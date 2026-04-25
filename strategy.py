#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_WithRegime
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume spike, and choppiness regime filter.
Long when price breaks above R1, 1d EMA34 trend bullish, volume spike, and choppy market (CHOP > 61.8).
Short when price breaks below S1, 1d EMA34 trend bearish, volume spike, and choppy market.
Exit when price re-enters H3/L3 range or 1d trend reverses.
Uses discrete position sizing 0.30 to limit fee drag. Target 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar's OHLC
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12  # R1 level
    s1 = prev_close_1d - 1.1 * camarilla_range / 12  # S1 level
    h3 = prev_close_1d + 1.1 * camarilla_range / 6   # H3 level
    l3 = prev_close_1d - 1.1 * camarilla_range / 6   # L3 level
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 61.8 = choppy/range (favor mean reversion/breakout)
    # CHOP < 38.2 = trending (favor trend following)
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_regime = chop > 61.8  # choppy market regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34), volume MA (20), and CHOP (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 1d trend bullish AND volume spike AND choppy regime
            long_setup = (close[i] > r1_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i] and \
                         chop_regime[i]
            # Short: price breaks below S1 AND 1d trend bearish AND volume spike AND choppy regime
            short_setup = (close[i] < s1_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i] and \
                          chop_regime[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price re-enters Camarilla H3/L3 range OR 1d trend turns bearish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price re-enters Camarilla H3/L3 range OR 1d trend turns bullish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_WithRegime"
timeframe = "4h"
leverage = 1.0