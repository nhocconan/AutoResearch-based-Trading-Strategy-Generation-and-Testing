#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike
# Uses 12h primary timeframe with 1d HTF for trend and Donchian calculation
# Donchian breakouts capture strong momentum when aligned with daily trend
# Volume confirmation filters false breakouts
# ATR-based stoploss manages risk
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull and bear markets by following 1d trend while capturing 12h momentum

name = "12h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels from prior 1d (using daily data)
    # Prior day's high/low for Donchian calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    # We use the completed prior 1d bar's rolling window
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (20*12h = ~10 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss calculation on 12h timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, Donchian, and ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50 = ema50_aligned[i]
        curr_donchian_upper = donchian_upper_aligned[i]
        curr_donchian_lower = donchian_lower_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr[i]
        
        # Trend regime: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: break above Donchian upper AND bullish regime
                if curr_high > curr_donchian_upper and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below Donchian lower AND bearish regime
                elif curr_low < curr_donchian_lower and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below Donchian lower (reversal) OR regime changes to bearish OR stoploss hit
            stoploss_level = entry_price - 2.5 * curr_atr
            if curr_low < curr_donchian_lower or not is_bullish_regime or curr_low < stoploss_level:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above Donchian upper (reversal) OR regime changes to bullish OR stoploss hit
            stoploss_level = entry_price + 2.5 * curr_atr
            if curr_high > curr_donchian_upper or not is_bearish_regime or curr_high > stoploss_level:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals