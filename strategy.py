#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter
# Long: Alligator bullish (jaw < teeth < lips) AND Elder Bull Power > 0 AND price > 1d EMA50
# Short: Alligator bearish (jaw > teeth > lips) AND Elder Bear Power < 0 AND price < 1d EMA50
# Exit: Alligator conflict (jaws cross teeth) OR Elder Power reverses OR price crosses 1d EMA50
# Using 1d EMA50 for higher timeframe trend alignment reduces whipsaws in 6h charts
# Williams Alligator identifies trend initiation/continuation via smoothed medians
# Elder Ray measures bull/bear power behind price movements
# Discrete position sizing: 0.25 for long/short to balance return and drawdown
# Target: 80-160 total trades over 4 years (20-40/year) on 6h timeframe

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: jaw (13,8), teeth (8,5), lips (5,3) - all SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line  
    lips = smma(close, 5)   # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 13)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any indicator is not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions: Alligator conflict (jaws cross teeth) OR Bear Power > 0 OR price < 1d EMA50
            if (jaw[i] > teeth[i]) or (curr_bear_power > 0) or (curr_close < curr_ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Alligator conflict (jaws cross teeth) OR Bull Power < 0 OR price > 1d EMA50
            if (jaw[i] < teeth[i]) or (curr_bull_power < 0) or (curr_close > curr_ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Alligator bullish (jaw < teeth < lips) AND Bull Power > 0 AND price > 1d EMA50
            if (jaw[i] < teeth[i] < lips[i]) and (curr_bull_power > 0) and (curr_close > curr_ema_1d):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Alligator bearish (jaw > teeth > lips) AND Bear Power < 0 AND price < 1d EMA50
            elif (jaw[i] > teeth[i] > lips[i]) and (curr_bear_power < 0) and (curr_close < curr_ema_1d):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals