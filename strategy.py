#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band AND close > 1d EMA200 AND volume > 2.0x 20-period average.
Short when price breaks below 20-period Donchian lower band AND close < 1d EMA200 AND volume > 2.0x 20-period average.
Exit when price crosses the opposite Donchian band (mean reversion) OR ATR-based stoploss hit (2.5 * ATR).
Uses 1d HTF for trend filter to improve robustness in both bull and bear markets.
Target: 12-25 trades/year per symbol to minimize fee drag while maintaining edge.
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
    
    # Get 12h data for Donchian calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian channels (20-period) on 12h
    def calculate_donchian(high_arr, low_arr, window):
        """Donchian channels: upper = max(high, window), lower = min(low, window)"""
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
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
    
    # Get Donchian levels (20-period)
    upper_20, lower_20 = calculate_donchian(high_12h, low_12h, 20)
    
    # Get 1d data for EMA200 trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss on 12h
    atr = calculate_atr(high_12h, low_12h, close_12h, 14)
    
    # Align all indicators to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        ema200 = ema200_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above upper band + price > 1d EMA200 + volume spike
            if price > upper and price > ema200 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Breakout below lower band + price < 1d EMA200 + volume spike
            elif price < lower and price < ema200 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price crosses below lower band (mean reversion)
            if price < lower:
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
            
            # Exit 1: Price crosses above upper band (mean reversion)
            if price > upper:
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

name = "12h_Donchian20_1dEMA200_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0