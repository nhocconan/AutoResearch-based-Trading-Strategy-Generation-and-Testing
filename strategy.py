#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with daily EMA34 filter and volume spike confirmation.
# Long when: Price above Alligator teeth (green line), jaws (blue) < teeth (green) < lips (red), daily EMA34 upward, volume > 1.5x 20-period average
# Short when: Price below Alligator teeth (green line), jaws (blue) > teeth (green) > lips (red), daily EMA34 downward, volume > 1.5x 20-period average
# Exit when: Price crosses back through the Alligator teeth (green line)
# Alligator identifies trend direction and alignment, EMA34 filters trend strength, volume confirms momentum.
# Target: 12-30 trades/year per symbol. Works in bull (buy alignments) and bear (sell alignments).
name = "12h_WilliamsAlligator_EMA34_Volume"
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
    
    # 1-day data for Williams Alligator and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator lines (13, 8, 5 SMAs shifted forward)
    # Jaw (blue): 13-period SMMA of median price, shifted 8 bars forward
    median_price_1d = (high_1d + low_1d) / 2.0
    sma13_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw_1d = np.roll(sma13_1d, 8)  # shift forward 8 bars
    jaw_1d[:8] = np.nan  # first 8 values invalid
    
    # Teeth (green): 8-period SMMA of median price, shifted 5 bars forward
    sma8_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth_1d = np.roll(sma8_1d, 5)  # shift forward 5 bars
    teeth_1d[:5] = np.nan  # first 5 values invalid
    
    # Lips (red): 5-period SMMA of median price, shifted 3 bars forward
    sma5_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips_1d = np.roll(sma5_1d, 3)  # shift forward 3 bars
    lips_1d[:3] = np.nan  # first 3 values invalid
    
    # Calculate EMA34 on daily data for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1D data to 12H timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price above teeth, jaws < teeth < lips (bullish alignment), EMA34 upward, volume spike
            if (price > teeth and jaw < teeth and teeth < lips and 
                ema34 > ema34_1d_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price below teeth, jaws > teeth > lips (bearish alignment), EMA34 downward, volume spike
            elif (price < teeth and jaw > teeth and teeth > lips and 
                  ema34 < ema34_1d_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below teeth
            if price < teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above teeth
            if price > teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals