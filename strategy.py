#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation.
# Long when: Jaw < Teeth < Lips (bullish alignment) with price > Lips, volume > 1.5x 20-period average, and price > 1d EMA50
# Short when: Jaw > Teeth > Lips (bearish alignment) with price < Lips, volume > 1.5x 20-period average, and price < 1d EMA50
# Exit when alignment breaks or price crosses Jaw.
# Williams Alligator identifies trend phases; EMA50 filters for higher timeframe trend; volume confirms strength.
# Designed for ~15-25 trades/year per symbol to avoid fee drag.
name = "12h_WilliamsAlligator_EMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator components (using median price)
    median_price = (high + low) / 2
    
    # Jaw (blue line): 13-period SMMA, shifted 8 bars ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars ahead
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth (red line): 8-period SMMA, shifted 5 bars ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars ahead
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips (green line): 5-period SMMA, shifted 3 bars ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars ahead
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': median_price}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': median_price}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': median_price}), lips)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Check Alligator alignment
        bullish_alignment = jaw_val < teeth_val < lips_val
        bearish_alignment = jaw_val > teeth_val > lips_val
        
        if position == 0:
            # Long entry: bullish alignment with price above lips, volume confirmation, and uptrend
            if bullish_alignment and price > lips_val and vol > 1.5 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment with price below lips, volume confirmation, and downtrend
            elif bearish_alignment and price < lips_val and vol > 1.5 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: alignment breaks or price crosses below jaw
            if not bullish_alignment or price < jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: alignment breaks or price crosses above jaw
            if not bearish_alignment or price > jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals