#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Camarilla pivot levels with price action confirmation.
# Long when price breaks above R4 with close > open (bullish candle) and volume > 1.3x average.
# Short when price breaks below S4 with close < open (bearish candle) and volume > 1.3x average.
# Exit when price returns to the weekly pivot point (PP).
# Uses weekly Camarilla levels for institutional support/resistance, price action for confirmation,
# and volume filter to avoid false breakouts. Works in trending markets (breakouts) and ranges (mean reversion to PP).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    # Based on previous week's high, low, close
    camarilla_pp = np.full(len(close_1w), np.nan)
    camarilla_r4 = np.full(len(close_1w), np.nan)
    camarilla_s4 = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        # Previous week's values
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        # Pivot point
        camarilla_pp[i] = (ph + pl + pc) / 3
        
        # Range
        rng = ph - pl
        
        # Camarilla levels
        camarilla_r4[i] = pc + rng * 1.1 / 2
        camarilla_s4[i] = pc - rng * 1.1 / 2
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly Camarilla (need at least 1 week of history) and volume MA
    start_idx = max(19, 1)  # volume MA20 and weekly data (need 1 week back)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.3 * vol_avg
        
        # Price action: bullish/bearish candle
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # Long: break above R4 with bullish candle and volume
            if (price > camarilla_r4_aligned[i] and 
                bullish_candle and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below S4 with bearish candle and volume
            elif (price < camarilla_s4_aligned[i] and 
                  bearish_candle and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot point
            if price <= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to weekly pivot point
            if price >= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyCamarilla_R4S4_Breakout_PP"
timeframe = "6h"
leverage = 1.0