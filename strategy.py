#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_chop_v1
# Hypothesis: 1d Donchian channel breakout with weekly trend filter, volume confirmation, and chop regime avoidance.
# Long: price breaks above Donchian(20) high AND weekly EMA(21) rising AND volume > 1.5x average AND chop < 61.8
# Short: price breaks below Donchian(20) low AND weekly EMA(21) falling AND volume > 1.5x average AND chop < 61.8
# Exit: opposite Donchian breakout or chop > 61.8 (range) or volume drops
# Designed to capture strong trends in both bull and bear markets while avoiding false breakouts in choppy/ranging conditions.
# Weekly EMA provides multi-timeframe trend direction, volume confirms institutional participation, chop filter avoids whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Weekly EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Weekly EMA slope (rising/falling)
    ema_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema_21_1w_aligned[i]) and not np.isnan(ema_21_1w_aligned[i-1]):
            ema_slope[i] = ema_21_1w_aligned[i] - ema_21_1w_aligned[i-1]
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Choppiness Index (14-period) - measures ranging vs trending
    chop = np.full(n, np.nan)
    for i in range(14, n):
        atr_sum = 0.0
        for j in range(i-13, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        if atr_sum > 0:
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_slope[i]) or np.isnan(avg_volume[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / avg_volume[i] if avg_volume[i] > 0 else 0
        
        # Long conditions: breakout above Donchian high + weekly uptrend + volume spike + not choppy
        long_breakout = price > donch_high[i]
        long_trend = ema_slope[i] > 0  # Weekly EMA rising
        long_volume = vol_ratio > 1.5  # Volume > 1.5x average
        long_chop = chop[i] < 61.8  # Not in choppy regime (trending)
        
        # Short conditions: breakout below Donchian low + weekly downtrend + volume spike + not choppy
        short_breakout = price < donch_low[i]
        short_trend = ema_slope[i] < 0  # Weekly EMA falling
        short_volume = vol_ratio > 1.5  # Volume > 1.5x average
        short_chop = chop[i] < 61.8  # Not in choppy regime (trending)
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR chop increases (range) OR volume drops significantly
            if price < donch_low[i] or chop[i] > 61.8 or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR chop increases (range) OR volume drops significantly
            if price > donch_high[i] or chop[i] > 61.8 or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish breakout with confirmation
            if long_breakout and long_trend and long_volume and long_chop:
                position = 1
                signals[i] = 0.25
            # Enter short: bearish breakout with confirmation
            elif short_breakout and short_trend and short_volume and short_chop:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals