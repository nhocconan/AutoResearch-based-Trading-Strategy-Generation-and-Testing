#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 reversal with 1d EMA50 trend filter and volume confirmation.
Long when price crosses above Camarilla S3 AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when price crosses below Camarilla R3 AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when price crosses Camarilla H4/L4 levels (continuation) OR ATR-based stoploss hit (2.5 * ATR).
Uses 1d HTF for trend filter to improve robustness in both bull and bear markets.
Target: 12-30 trades/year per symbol to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Get 1d data for EMA50 trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 6h bar's range
    def calculate_camarilla(high_arr, low_arr, close_arr):
        """Calculate Camarilla pivot levels for given HLC arrays"""
        # Camarilla uses previous bar's close and range
        rng = high_arr - low_arr
        c = close_arr
        
        # Initialize arrays
        h6 = np.full_like(high_arr, np.nan)
        h5 = np.full_like(high_arr, np.nan)
        h4 = np.full_like(high_arr, np.nan)
        h3 = np.full_like(high_arr, np.nan)
        l3 = np.full_like(low_arr, np.nan)
        l4 = np.full_like(low_arr, np.nan)
        l5 = np.full_like(low_arr, np.nan)
        l6 = np.full_like(low_arr, np.nan)
        
        # Calculate for each bar (starting from index 1 as we need previous bar)
        for i in range(1, len(high_arr)):
            # Use previous bar's data
            prev_high = high_arr[i-1]
            prev_low = low_arr[i-1]
            prev_close = close_arr[i-1]
            prev_rng = prev_high - prev_low
            
            # Camarilla formulas
            h6[i] = prev_close + 1.1 * prev_rng
            h5[i] = prev_close + 1.1 * prev_rng / 2
            h4[i] = prev_close + 1.1 * prev_rng / 4
            h3[i] = prev_close + 1.1 * prev_rng / 6
            l3[i] = prev_close - 1.1 * prev_rng / 6
            l4[i] = prev_close - 1.1 * prev_rng / 4
            l5[i] = prev_close - 1.1 * prev_rng / 2
            l6[i] = prev_close - 1.1 * prev_rng
        
        return h6, h5, h4, h3, l3, l4, l5, l6
    
    # Calculate ATR (14-period) for stoploss on 6h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    # Get Camarilla levels
    h6, h5, h4, h3, l3, l4, l5, l6 = calculate_camarilla(high_6h, low_6h, close_6h)
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 6h
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss on 6h
    atr = calculate_atr(high_6h, low_6h, close_6h, 14)
    
    # Align all indicators to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_6h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_6h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_6h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_6h, l4)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        ema50 = ema50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Reversal above S3 + price > 1d EMA50 + volume confirmation
            if price > l3 and price > ema50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Reversal below R3 + price < 1d EMA50 + volume confirmation
            elif price < h3 and price < ema50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price crosses above H4 (continuation - take profit)
            if price > h4:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR below entry)
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Price crosses below L4 (continuation - take profit)
            if price < l4:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR above entry)
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA50_VolumeReversal_ATRStop"
timeframe = "6h"
leverage = 1.0