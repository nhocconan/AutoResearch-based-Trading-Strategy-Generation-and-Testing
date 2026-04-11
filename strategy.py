#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_williams_alligator_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator components on daily timeframe
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    smma_jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(smma_jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    smma_teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(smma_teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    smma_lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(smma_lips, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 4h timeframe
    jaw_4h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_4h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_4h = align_htf_to_ltf(prices, df_1d, lips)
    
    # 4h ATR for volatility filter (21 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    # 4h volume filter: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_4h[i]) or np.isnan(teeth_4h[i]) or np.isnan(lips_4h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_30[i]
        
        # Volume confirmation (1.8x average)
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Williams Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        bullish_alignment = lips_4h[i] > teeth_4h[i] and teeth_4h[i] > jaw_4h[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips_4h[i] < teeth_4h[i] and teeth_4h[i] < jaw_4h[i]
        
        # Long conditions: bullish alignment + price above lips + volume confirmation
        long_signal = volume_confirmed and bullish_alignment and (price_close > lips_4h[i])
        
        # Short conditions: bearish alignment + price below lips + volume confirmation
        short_signal = volume_confirmed and bearish_alignment and (price_close < lips_4h[i])
        
        # Exit when Alligator lines intertwine (no clear alignment)
        no_alignment = not (bullish_alignment or bearish_alignment)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and no_alignment:
            position = 0
            signals[i] = 0.0
        elif position == -1 and no_alignment:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Williams Alligator trend-following strategy on 4h timeframe with daily Alligator alignment.
# Enters long when daily Williams Alligator shows bullish alignment (Lips > Teeth > Jaw) 
# and 4h price closes above the Lips line with volume >1.8x average.
# Enters short when bearish alignment (Lips < Teeth < Jaw) and price closes below Lips with volume confirmation.
# Exits when Alligator lines intertwine (no clear alignment), indicating trend weakness or consolidation.
# The Alligator's smoothed moving averages filter out noise and catch sustained trends in both bull and bear markets.
# Volume confirmation ensures participation, reducing false signals.
# Target: 20-40 trades per year to minimize fee drag while capturing strong trends.
# Williams Alligator is effective in trending markets and avoids whipsaws during ranging periods.