#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Strong bull power + ADX > 25 + volume spike = long entry; strong bear power + ADX > 25 + volume spike = short
# Works in bull markets via strong bear power reversals and in bear markets via strong bull power reversals
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1dADX25_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    # Avoid division by zero
    atr_1d_safe = np.where(atr_1d == 0, 1e-10, atr_1d)
    plus_di_1d = 100 * wilder_smooth(plus_dm, period) / atr_1d_safe
    minus_di_1d = 100 * wilder_smooth(minus_dm, period) / atr_1d_safe
    dx_denom = plus_di_1d + minus_di_1d
    dx_denom_safe = np.where(dx_denom == 0, 1e-10, dx_denom)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / dx_denom_safe
    adx_1d = wilder_smooth(dx_1d, period)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    # Volume confirmation: volume > 2.0x 96-period average (96*6h = 576h = 24 days)
    vol_ma_96 = pd.Series(volume).rolling(window=96, min_periods=96).mean().values
    volume_spike = volume > (2.0 * vol_ma_96)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 96)  # warmup for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema13_1d_aligned[i]) or np.isnan(ema13_6h[i]) or 
            np.isnan(vol_ma_96[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_adx = adx_1d_aligned[i]
        curr_ema13_1d = ema13_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and strong trend (ADX > 25)
            if curr_volume_spike and curr_adx > 25:
                # Bullish entry: strong bear power (price below EMA13) with reversal signs
                # Bear power negative and increasing (less negative) = bullish reversal
                if curr_bear_power < 0 and i > start_idx and curr_bear_power > bear_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: strong bull power (price above EMA13) with reversal signs
                # Bull power positive and decreasing = bearish reversal
                elif curr_bull_power > 0 and i > start_idx and curr_bull_power < bull_power[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when bear power becomes strongly negative again OR trend weakens (ADX < 20)
            if curr_bear_power < -0.5 * np.std(bull_power[max(0, i-50):i+1]) or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when bull power becomes strongly positive again OR trend weakens (ADX < 20)
            if curr_bull_power > 0.5 * np.std(bear_power[max(0, i-50):i+1]) or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals