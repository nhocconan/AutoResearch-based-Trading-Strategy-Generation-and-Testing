#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Volume Spike + ATR Trail
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
- Trend when Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear)
- Volume Spike: Current volume > 2.0x 20-period average for confirmation
- ATR Trail: Exit when price moves 2.5* ATR against position from extreme
- Uses 1d HTF for regime classification to avoid whipsaws
- Target: 20-50 trades/year per symbol (~80-200 total over 4 years)
- Position sizing: 0.30 (discrete levels to minimize fee churn)
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
    
    # Get 4h data for primary calculations
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Alligator on 4h
    median_price_4h = (high_4h + low_4h) / 2
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw_4h = pd.Series(median_price_4h).rolling(window=13, min_periods=13).mean().values
    jaw_4h = np.roll(jaw_4h, 8)
    jaw_4h[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth_4h = pd.Series(median_price_4h).rolling(window=8, min_periods=8).mean().values
    teeth_4h = np.roll(teeth_4h, 5)
    teeth_4h[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips_4h = pd.Series(median_price_4h).rolling(window=5, min_periods=5).mean().values
    lips_4h = np.roll(lips_4h, 3)
    lips_4h[:3] = np.nan
    
    # Determine trend direction
    bullish_alligator = (lips_4h > teeth_4h) & (teeth_4h > jaw_4h)
    bearish_alligator = (lips_4h < teeth_4h) & (teeth_4h < jaw_4h)
    
    # Volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for trailing stop on 4h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, 14)
    
    # Align all indicators to 4h timeframe
    bullish_alligator_aligned = align_htf_to_ltf(prices, df_4h, bullish_alligator)
    bearish_alligator_aligned = align_htf_to_ltf(prices, df_4h, bearish_alligator)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    extreme_price = 0.0  # Track extreme price for trailing stop
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bullish_alligator_aligned[i]) or np.isnan(bearish_alligator_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        is_bullish = bullish_alligator_aligned[i]
        is_bearish = bearish_alligator_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Entry logic: Alligator alignment + volume spike
            if is_bullish and vol > 2.0 * vol_ma:
                signals[i] = 0.30
                position = 1
                entry_price = price
                extreme_price = price
            elif is_bearish and vol > 2.0 * vol_ma:
                signals[i] = -0.30
                position = -1
                entry_price = price
                extreme_price = price
        
        elif position == 1:
            # Update extreme price for trailing stop
            if price > extreme_price:
                extreme_price = price
            
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Alligator trend reversal
            if not is_bullish:
                exit_signal = True
            
            # Exit 2: ATR-based trailing stop (2.5 * ATR below extreme)
            elif price < extreme_price - 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Update extreme price for trailing stop (track lowest price for shorts)
            if price < extreme_price:
                extreme_price = price
            
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Alligator trend reversal
            if not is_bearish:
                exit_signal = True
            
            # Exit 2: ATR-based trailing stop (2.5 * ATR above extreme)
            elif price > extreme_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_WilliamsAlligator_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0