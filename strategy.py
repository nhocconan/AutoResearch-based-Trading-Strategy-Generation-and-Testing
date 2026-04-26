#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ChopFilter
Hypothesis: On 12h timeframe, price breaking Camarilla R1/S1 levels with 1w EMA50 trend alignment, volume confirmation (2.0x), and chop regime filter (CHOP > 61.8 for mean reversion avoidance) provides robust breakout signals. Uses discrete sizing (±0.25) and ATR stoploss (2.5x) to control risk. Targets 12-37 trades/year (50-150 over 4 years) to stay within optimal trade frequency for 12h timeframe. Designed to work in both bull and bear markets by requiring strong trend alignment and volume conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Choppiness Index (CHOP) on 12h for regime filter
    def calculate_chop(high, low, close, window=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum()
        hh = pd.Series(high).rolling(window=window, min_periods=window).max()
        ll = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window)
        return chop.values
    
    chop = calculate_chop(high, low, close, window=14)
    chop_filter = chop > 61.8  # Only trade in choppy/range markets (avoid strong trends)
    
    # Calculate Camarilla levels from previous 12h bar
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1w EMA(50), ATR(14), volume MA(20), CHOP(14)
    start_idx = max(50, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(chop[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average
        trend_1w_up = close_val > ema_50_1w_aligned[i]
        trend_1w_down = close_val < ema_50_1w_aligned[i]
        in_chop_zone = chop_filter[i]  # Only trade when CHOP > 61.8 (range-bound)
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND 1w trend up AND volume confirmation AND in chop zone
            long_signal = (close_val > camarilla_r1[i]) and trend_1w_up and vol_confirmed and in_chop_zone
            
            # Short: price breaks below Camarilla S1 AND 1w trend down AND volume confirmation AND in chop zone
            short_signal = (close_val < camarilla_s1[i]) and trend_1w_down and vol_confirmed and in_chop_zone
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price hits ATR stoploss OR chop regime breaks down (CHOP < 38.2 = strong trend)
            if (not trend_1w_up) or (close_val < entry_price - 2.5 * atr[i]) or (chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss OR chop regime breaks down (CHOP < 38.2 = strong trend)
            if (not trend_1w_down) or (close_val > entry_price + 2.5 * atr[i]) or (chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0