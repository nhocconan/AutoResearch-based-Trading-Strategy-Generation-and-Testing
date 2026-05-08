#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with weekly trend filter and volume spike
# Long when price > Alligator Jaw, weekly EMA(50) uptrend, and volume spike
# Short when price < Alligator Jaw, weekly EMA(50) downtrend, and volume spike
# Alligator uses SMAs: Jaw (13), Teeth (8), Lips (5) - smoothed
# Weekly EMA filters for higher timeframe trend alignment
# Volume spike confirms institutional participation; avoids false breakouts
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "12h_WilliamsAlligator_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Alligator and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator from weekly data
    weekly_close = df_1w['close'].values
    # Jaw: 13-period SMMA (smoothed moving average)
    jaw = pd.Series(weekly_close).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMMA
    teeth = pd.Series(weekly_close).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMMA
    lips = pd.Series(weekly_close).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Weekly EMA(50) for trend filter
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: current volume > 2.0 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        price = close[i]
        vol_spike = volume_spike[i]
        
        # Alligator condition: price > Jaw for long, price < Jaw for short
        # Additional confirmation: Teeth > Lips for bullish alignment, Teeth < Lips for bearish
        if position == 0:
            # Enter long: price > Jaw, teeth > lips (bullish alignment), weekly uptrend, volume spike
            if price > jaw_val and teeth_val > lips_val and price > ema50_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Jaw, teeth < lips (bearish alignment), weekly downtrend, volume spike
            elif price < jaw_val and teeth_val < lips_val and price < ema50_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Jaw or teeth < lips (loss of bullish alignment)
            if price < jaw_val or teeth_val < lips_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Jaw or teeth > lips (loss of bearish alignment)
            if price > jaw_val or teeth_val > lips_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals