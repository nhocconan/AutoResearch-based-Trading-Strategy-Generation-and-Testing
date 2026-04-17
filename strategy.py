#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND close > 12h EMA34 AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S1 AND close < 12h EMA34 AND volume > 1.8x 20-period average.
Exit when price crosses Camarilla H3/L3 levels (mean reversion) OR ATR-based stoploss hit (2.0 * ATR).
Uses 12h HTF for trend filter to improve robustness in both bull and bear markets.
Target: 20-40 trades/year per symbol to minimize fee drag while maintaining edge.
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
    
    # Get 4h data for Camarilla calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels from previous 4h bar's range
    def calculate_camarilla(high_arr, low_arr, close_arr):
        """Calculate Camarilla pivot levels for given HLC arrays"""
        # Camarilla uses previous bar's close and range
        rng = high_arr - low_arr
        c = close_arr
        
        # Initialize arrays
        h5 = np.full_like(high_arr, np.nan)
        h4 = np.full_like(high_arr, np.nan)
        h3 = np.full_like(high_arr, np.nan)
        l3 = np.full_like(low_arr, np.nan)
        l4 = np.full_like(low_arr, np.nan)
        l5 = np.full_like(low_arr, np.nan)
        
        # Calculate for each bar (starting from index 1 as we need previous bar)
        for i in range(1, len(high_arr)):
            # Use previous bar's data
            prev_high = high_arr[i-1]
            prev_low = low_arr[i-1]
            prev_close = close_arr[i-1]
            prev_rng = prev_high - prev_low
            
            # Camarilla formulas
            h5[i] = prev_close + 1.1 * prev_rng / 2
            h4[i] = prev_close + 1.1 * prev_rng / 4
            h3[i] = prev_close + 1.1 * prev_rng / 6
            l3[i] = prev_close - 1.1 * prev_rng / 6
            l4[i] = prev_close - 1.1 * prev_rng / 4
            l5[i] = prev_close - 1.1 * prev_rng / 2
        
        return h5, h4, h3, l3, l4, l5
    
    # Calculate ATR (14-period) for stoploss on 4h
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
    h5, h4, h3, l3, l4, l5 = calculate_camarilla(high_4h, low_4h, close_4h)
    
    # Get 12h data for EMA34 trend filter (higher timeframe)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss on 4h
    atr = calculate_atr(high_4h, low_4h, close_4h, 14)
    
    # Align all indicators to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 35  # warmup for EMA34 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        ema34 = ema34_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above H3 + price > 12h EMA34 + volume spike
            if price > h3 and price > ema34 and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Breakout below L3 + price < 12h EMA34 + volume spike
            elif price < l3 and price < ema34 and vol > 1.8 * vol_ma:
                signals[i] = -0.25
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
                signals[i] = 0.25
        
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
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0