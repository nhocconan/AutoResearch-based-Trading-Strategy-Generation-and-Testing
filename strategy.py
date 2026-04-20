#!/usr/bin/env python3
# 6h_1d_OrderBlock_Confluence
# Hypothesis: Identify institutional order blocks from daily structure and trade 6h retracements into these zones with volume confirmation.
# Uses weekly trend filter to ensure trades align with higher timeframe momentum.
# Works in bull/bear: In uptrend, look for bullish order blocks (demand zones); in downtrend, bearish order blocks (supply zones).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_OrderBlock_Confluence"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for structure and order blocks
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Daily structure: Identify swing points for order blocks ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Find swing highs and lows (3-bar lookback)
    swing_high = np.zeros(len(high_1d), dtype=bool)
    swing_low = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d)-2):
        # Swing high: higher than 2 bars on each side
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = True
        # Swing low: lower than 2 bars on each side
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = True
    
    # Identify bullish and bearish order blocks
    # Bullish OB: last down candle before a strong up move (from swing low to swing high)
    # Bearish OB: last up candle before a strong down move (from swing high to swing low)
    bullish_ob_low = np.full(len(high_1d), np.nan)
    bullish_ob_high = np.full(len(high_1d), np.nan)
    bearish_ob_low = np.full(len(high_1d), np.nan)
    bearish_ob_high = np.full(len(high_1d), np.nan)
    
    # Track recent swing points
    last_swing_low_idx = -1
    last_swing_high_idx = -1
    
    for i in range(len(high_1d)):
        if swing_low[i]:
            last_swing_low_idx = i
            # Look for bullish OB: bearish candle before upswing
            if last_swing_high_idx != -1 and last_swing_high_idx < i:
                # Find the bearish candle between last swing high and this swing low
                for j in range(last_swing_high_idx, i):
                    if close_1d[j] < open_1d[j]:  # Bearish candle
                        bullish_ob_low[j] = low_1d[j]
                        bullish_ob_high[j] = high_1d[j]
                        break
        
        if swing_high[i]:
            last_swing_high_idx = i
            # Look for bearish OB: bullish candle before downswing
            if last_swing_low_idx != -1 and last_swing_low_idx < i:
                # Find the bullish candle between last swing low and this swing high
                for j in range(last_swing_low_idx, i):
                    if close_1d[j] > open_1d[j]:  # Bullish candle
                        bearish_ob_low[j] = low_1d[j]
                        bearish_ob_high[j] = high_1d[j]
                        break
    
    # Need open prices for OB detection
    open_1d = df_1d['open'].values
    
    # Re-run OB detection with proper open prices
    bullish_ob_low = np.full(len(high_1d), np.nan)
    bullish_ob_high = np.full(len(high_1d), np.nan)
    bearish_ob_low = np.full(len(high_1d), np.nan)
    bearish_ob_high = np.full(len(high_1d), np.nan)
    
    last_swing_low_idx = -1
    last_swing_high_idx = -1
    
    for i in range(len(high_1d)):
        if swing_low[i]:
            last_swing_low_idx = i
            if last_swing_high_idx != -1 and last_swing_high_idx < i:
                for j in range(last_swing_high_idx, i):
                    if close_1d[j] < open_1d[j]:  # Bearish candle
                        bullish_ob_low[j] = low_1d[j]
                        bullish_ob_high[j] = high_1d[j]
                        break
        
        if swing_high[i]:
            last_swing_high_idx = i
            if last_swing_low_idx != -1 and last_swing_low_idx < i:
                for j in range(last_swing_low_idx, i):
                    if close_1d[j] > open_1d[j]:  # Bullish candle
                        bearish_ob_low[j] = low_1d[j]
                        bearish_ob_high[j] = high_1d[j]
                        break
    
    # === Weekly EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 6h: Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all daily and weekly levels to 6h
    bullish_ob_low_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_low)
    bullish_ob_high_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_high)
    bearish_ob_low_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_low)
    bearish_ob_high_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_high)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ob_low_bull = bullish_ob_low_aligned[i]
        ob_high_bull = bullish_ob_high_aligned[i]
        ob_low_bear = bearish_ob_low_aligned[i]
        ob_high_bear = bearish_ob_high_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ob_low_bull) or np.isnan(ob_high_bull) or np.isnan(ob_low_bear) or 
            np.isnan(ob_high_bear) or np.isnan(ema34_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price retraces into bullish order block (demand zone) with volume confirmation and above weekly EMA34
            if (close_val >= ob_low_bull and close_val <= ob_high_bull and  # Inside OB
                vol_ratio_val > 1.8 and  # Volume confirmation
                close_val > ema34_1w_val):  # Only long in weekly uptrend
                signals[i] = 0.25
                position = 1
            # Short: Price retraces into bearish order block (supply zone) with volume confirmation and below weekly EMA34
            elif (close_val >= ob_low_bear and close_val <= ob_high_bear and  # Inside OB
                  vol_ratio_val > 1.8 and  # Volume confirmation
                  close_val < ema34_1w_val):  # Only short in weekly downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price reaches OB high or shows weakness
            if close_val >= ob_high_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price reaches OB low or shows weakness
            if close_val <= ob_low_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals