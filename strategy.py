#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high AND close > 1w EMA50 AND volume > 1.5x 20-day average.
Short when price breaks below 20-day low AND close < 1w EMA50 AND volume > 1.5x 20-day average.
Exit when price crosses opposite Donchian level (mean reversion) OR ATR-based stoploss hit (2.0 * ATR).
Uses 1w HTF for trend filter to improve robustness in both bull and bear markets.
Target: 15-25 trades/year per symbol to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA50 trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    def calculate_donchian(high_arr, low_arr, window):
        """Donchian channels: upper = max(high, window), lower = min(low, window)"""
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    # Calculate EMA50 on 1w
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 1d
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss on 1d
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    # Get Donchian channels
    upper, lower = calculate_donchian(high_1d, low_1d, 20)
    
    # Align all indicators to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        up = upper_aligned[i]
        low = lower_aligned[i]
        ema50 = ema50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above upper band + price > 1w EMA50 + volume spike
            if price > up and price > ema50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Breakout below lower band + price < 1w EMA50 + volume spike
            elif price < low and price < ema50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price crosses below lower band (mean reversion)
            if price < low:
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
            
            # Exit 1: Price crosses above upper band (mean reversion)
            if price > up:
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

name = "1d_Donchian20_1wEMA50_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0