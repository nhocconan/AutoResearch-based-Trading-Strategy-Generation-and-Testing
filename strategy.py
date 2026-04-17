#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d EMA34 trend filter.
Long when price breaks above Camarilla R3 AND volume > 1.5x 20-period 12h average AND close > 1d EMA34.
Short when price breaks below Camarilla S3 AND volume > 1.5x 20-period 12h average AND close < 1d EMA34.
Exit when price crosses the Camarilla pivot point (PP) OR ATR-based stoploss hit (2.0 * ATR).
Uses 12h HTF for volume confirmation and 1d HTF for trend filter to improve robustness.
Target: 12-37 trades/year per symbol to avoid fee drag on 6h timeframe.
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
    
    # Get 12h data for Camarilla pivot and volume confirmation (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla pivot levels on 12h
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    # Range = High - Low
    range_12h = high_12h - low_12h
    # Resistance levels
    r1_12h = pp_12h + (range_12h * 1.0 / 12.0)
    r2_12h = pp_12h + (range_12h * 2.0 / 12.0)
    r3_12h = pp_12h + (range_12h * 3.0 / 12.0)
    r4_12h = pp_12h + (range_12h * 4.0 / 12.0)
    # Support levels
    s1_12h = pp_12h - (range_12h * 1.0 / 12.0)
    s2_12h = pp_12h - (range_12h * 2.0 / 12.0)
    s3_12h = pp_12h - (range_12h * 3.0 / 12.0)
    s4_12h = pp_12h - (range_12h * 4.0 / 12.0)
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume average (20-period) on 12h for confirmation
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss on 12h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_12h = calculate_atr(high_12h, low_12h, close_12h, 14)
    
    # Align all indicators to 6h timeframe (primary)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        pp = pp_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        ema34 = ema34_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above R3 + volume confirmation + price > 1d EMA34
            if price > r3 and vol > 1.5 * vol_ma and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Breakout below S3 + volume confirmation + price < 1d EMA34
            elif price < s3 and vol > 1.5 * vol_ma and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price crosses pivot point (mean reversion to PP)
            if price < pp:
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
            
            # Exit 1: Price crosses pivot point (mean reversion to PP)
            if price > pp:
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

name = "6h_Camarilla_R3S3_12hVolume_1dEMA34_ATRStop"
timeframe = "6h"
leverage = 1.0