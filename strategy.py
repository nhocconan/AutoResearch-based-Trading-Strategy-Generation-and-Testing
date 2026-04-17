#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1 AND close > 4h EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below S1 AND close < 4h EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses the Camarilla pivot point (mean) OR ATR-based stoploss hit.
Uses 4h HTF for trend filter to reduce false signals and improve Sharpe in both bull and bear markets.
Target: 15-37 trades/year (60-150 over 4 years) for 1h timeframe with session filter (08-20 UTC).
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
    df_1h = prices.copy()
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Camarilla levels (based on previous bar)
    def calculate_camarilla(high_arr, low_arr, close_arr):
        """Calculate Camarilla pivot levels"""
        pivot = (high_arr + low_arr + close_arr) / 3.0
        range_val = high_arr - low_arr
        r1 = pivot + (range_val * 1.1 / 12)
        s1 = pivot - (range_val * 1.1 / 12)
        return pivot, r1, s1
    
    # Calculate for previous bar (to avoid look-ahead)
    pivot = np.full_like(close_1h, np.nan, dtype=float)
    r1 = np.full_like(close_1h, np.nan, dtype=float)
    s1 = np.full_like(close_1h, np.nan, dtype=float)
    
    for i in range(1, len(close_1h)):
        p, r, s = calculate_camarilla(high_1h[i-1], low_1h[i-1], close_1h[i-1])
        pivot[i] = p
        r1[i] = r
        s1[i] = s
    
    # Get 4h data for EMA50 trend filter (higher timeframe)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 1h
    volume_1h = df_1h['volume'].values
    volume_1h_series = pd.Series(volume_1h)
    volume_ma_1h = volume_1h_series.rolling(window=20, min_periods=20).mean().values
    
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
    
    atr = calculate_atr(high_1h, low_1h, close_1h, 14)
    
    # Align all indicators to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1h, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1h, s1)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1h, volume_ma_1h)
    atr_aligned = align_htf_to_ltf(prices, df_1h, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 50  # warmup for indicators
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema50 = ema50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above R1 + price > 4h EMA50 + volume spike
            if price > r1_val and price > ema50 and vol > 2.0 * vol_ma:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Breakout below S1 + price < 4h EMA50 + volume spike
            elif price < s1_val and price < ema50 and vol > 2.0 * vol_ma:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price crosses pivot point (mean reversion)
            if price < pivot_val:
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
            
            # Exit 1: Price crosses pivot point (mean reversion)
            if price > pivot_val:
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

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_ATRStop"
timeframe = "1h"
leverage = 1.0