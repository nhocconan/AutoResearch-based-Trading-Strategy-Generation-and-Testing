#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# The Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) with future offsets.
# Enter long when Lips > Teeth > Jaw (bullish alignment) and price > Lips, with 1w close > 1w SMA(50) and volume expansion.
# Enter short when Lips < Teeth < Jaw (bearish alignment) and price < Lips, with 1w close < 1w SMA(50) and volume expansion.
# Uses 1d timeframe for Alligator calculation and 1w for trend filter.
# Designed for 15-35 trades/year on 1d timeframe with focus on trend alignment.
# Volume filter ensures breakouts have conviction, reducing false signals.
# 1w trend filter prevents counter-trend trading in choppy markets.

name = "1d_1w_alligator_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Alligator (already the base timeframe, but we use it for consistency)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Alligator components (SMAs with specific periods)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using SMA as approximation for SMMA (common in implementations)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Calculate 1w SMA(50) for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align 1w SMA to 1d timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after Jaw period (longest SMA)
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 1w trend direction
        is_uptrend = close[i] > sma_50_1w_aligned[i]
        is_downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Alligator alignment conditions
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Price relative to Lips (strongest signal)
        price_above_lips = close[i] > lips[i]
        price_below_lips = close[i] < lips[i]
        
        # Entry conditions
        bullish_entry = bullish_alignment and price_above_lips and vol_filter and is_uptrend
        bearish_entry = bearish_alignment and price_below_lips and vol_filter and is_downtrend
        
        # Exit conditions: opposite alignment
        exit_long = bearish_alignment  # Bearish alignment triggers long exit
        exit_short = bullish_alignment  # Bullish alignment triggers short exit
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals