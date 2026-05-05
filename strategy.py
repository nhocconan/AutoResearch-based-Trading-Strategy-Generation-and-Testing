#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) with volume confirmation and ATR-based stoploss
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND volume > 1.5 * avg_volume(20)
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND volume > 1.5 * avg_volume(20)
# Exit when Alligator alignment reverses OR price crosses the middle line (Teeth)
# Uses discrete sizing 0.30 to balance return and risk
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# Williams Alligator identifies trend strength and direction with built-in smoothing
# Volume confirmation ensures breakout validity and reduces false signals
# ATR stoploss manages risk during sideways markets
# Works in bull markets (strong uptrends with alignment) and bear markets (strong downtrends with alignment)

name = "4h_WilliamsAlligator_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need enough for Alligator (max period 13)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (SMAs of median price)
    # Median price = (High + Low) / 2
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Jaw: 13-period SMA, 8 bars offset
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, 5 bars offset  
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, 3 bars offset
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to 4h timeframe (wait for completed daily bar)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate ATR(14) for dynamic stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish alignment (Lips > Teeth > Jaw) AND price > Lips AND volume confirmation
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] and 
                close[i] > lips_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: Bearish alignment (Lips < Teeth < Jaw) AND price < Lips AND volume confirmation
            elif (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] and 
                  close[i] < lips_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Bearish alignment OR price crosses below Teeth (middle line) OR ATR-based stop
            if (lips_1d_aligned[i] < teeth_1d_aligned[i] or 
                close[i] < teeth_1d_aligned[i] or
                close[i] < prices['close'].iloc[i-1] - 2.0 * atr[i]):  # ATR stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Bullish alignment OR price crosses above Teeth (middle line) OR ATR-based stop
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] or 
                close[i] > teeth_1d_aligned[i] or
                close[i] > prices['close'].iloc[i-1] + 2.0 * atr[i]):  # ATR stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals