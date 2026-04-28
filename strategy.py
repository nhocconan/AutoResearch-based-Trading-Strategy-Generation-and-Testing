#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX25 Regime + Volume Spike
# Uses Elder Ray (Bull/Bear Power) to measure trend strength relative to EMA13.
# Regime filter: 1d ADX > 25 = trending (only trade with trend), ADX < 20 = ranging (fade extremes).
# Volume spike (>2.0x 20-bar avg) confirms momentum.
# In trending regime: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# In ranging regime: long at BB lower band, short at BB upper band (mean reversion).
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Works in both bull and bear via adaptive regime logic.

name = "6h_ElderRay_1dADX25_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for regime filter (ADX > 25 = trending, < 20 = ranging)
    # ADX calculation: +DM, -DM, TR, then DX, then ADX
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d[::-1])[::-1]  # negative of low diff
    low_diff = np.append(low_diff[1:], low_diff[-1])  # shift and append last
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period_adx = 14
    if len(plus_dm) >= period_adx:
        smoothed_plus_dm = wilders_smoothing(plus_dm, period_adx)
        smoothed_minus_dm = wilders_smoothing(minus_dm, period_adx)
        smoothed_tr = wilders_smoothing(tr, period_adx)
        
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = wilders_smoothing(dx, period_adx)
    else:
        adx = np.full_like(close_1d, np.nan)
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=0)
    
    # Calculate 6h Bollinger Bands for ranging regime (20, 2.0)
    close_s = pd.Series(close)
    bb_ma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std_20 = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma_20 + 2.0 * bb_std_20
    bb_lower = bb_ma_20 - 2.0 * bb_std_20
    
    # Calculate 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Need to align 6h high/low with 1d EMA13
    # For simplicity, use close price for EMA13 alignment approximation
    bull_power = high - ema_13_1d_aligned  # Bull Power: High - EMA13
    bear_power = low - ema_13_1d_aligned   # Bear Power: Low - EMA13
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(bb_ma_20[i]) or 
            np.isnan(bb_std_20[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d ADX > 25 = trending, ADX < 20 = ranging
        adx_val = adx_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if is_trending and vol_confirm:
            # Trending regime: trade with Elder Ray momentum
            long_signal = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
            short_signal = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
            
            # Exit on power reversal
            long_exit = bull_power[i] <= 0 or bull_power[i] < bull_power[i-1]
            short_exit = bear_power[i] >= 0 or bear_power[i] > bear_power[i-1]
            
            if long_signal and position <= 0:
                signals[i] = 0.25
                position = 1
            elif short_signal and position >= 0:
                signals[i] = -0.25
                position = -1
            elif position == 1 and long_exit:
                signals[i] = 0.0
                position = 0
            elif position == -1 and short_exit:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
                
        elif is_ranging:
            # Ranging regime: mean reversion at Bollinger Bands
            long_signal = close[i] <= bb_lower[i]
            short_signal = close[i] >= bb_upper[i]
            
            # Exit when price returns to mean
            long_exit = close[i] >= bb_ma_20[i]
            short_exit = close[i] <= bb_ma_20[i]
            
            if long_signal and position <= 0:
                signals[i] = 0.25
                position = 1
            elif short_signal and position >= 0:
                signals[i] = -0.25
                position = -1
            elif position == 1 and long_exit:
                signals[i] = 0.0
                position = 0
            elif position == -1 and short_exit:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # ADX between 20-25: no clear regime, stay flat
            signals[i] = 0.0
            position = 0
    
    return signals