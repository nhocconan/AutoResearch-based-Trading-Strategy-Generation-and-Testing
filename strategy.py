#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H4/L4 breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla H4 AND close > 4h EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla L4 AND close < 4h EMA50 AND volume > 1.8x 20-period average.
Exit when price crosses Camarilla H3/L3 levels (mean reversion) OR ATR-based stoploss hit (2.0 * ATR).
Uses 4h HTF for trend filter to improve robustness in both bull and bear markets.
Target: 15-37 trades/year per symbol to minimize fee drag while maintaining edge.
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
    
    # Get 1h data for Camarilla calculation (primary timeframe)
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Get 4h data for EMA50 trend filter (higher timeframe)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Camarilla levels from previous 1h bar's range
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
    
    # Calculate ATR (14-period) for stoploss on 1h
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
    h6, h5, h4, h3, l3, l4, l5, l6 = calculate_camarilla(high_1h, low_1h, close_1h)
    
    # Calculate EMA50 on 4h
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 1h
    volume_1h_series = pd.Series(volume_1h)
    volume_ma_1h = volume_1h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss on 1h
    atr = calculate_atr(high_1h, low_1h, close_1h, 14)
    
    # Align all indicators to 1h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1h, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1h, l3)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1h, volume_ma_1h)
    atr_aligned = align_htf_to_ltf(prices, df_1h, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        ema50 = ema50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above H4 + price > 4h EMA50 + volume spike
            if price > h4 and price > ema50 and vol > 1.8 * vol_ma:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Breakout below L4 + price < 4h EMA50 + volume spike
            elif price < l4 and price < ema50 and vol > 1.8 * vol_ma:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price crosses below L3 (mean reversion to lower level)
            if price < l3:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.0 * ATR below entry)
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Price crosses above H3 (mean reversion to upper level)
            if price > h3:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.0 * ATR above entry)
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H4L4_4hEMA50_VolumeSpike_ATRStop"
timeframe = "1h"
leverage = 1.0