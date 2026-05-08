#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Williams_Alligator_Trend_Momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Williams Alligator and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Williams Alligator on 1d: 3 SMAs ===
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, center=False, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, center=False, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, center=False, min_periods=5).mean().values
    
    # Align to 12h
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1d ATR for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_12h = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # === 12h Volume filter ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(atr10_12h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator conditions: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
            # Wait for clear alignment to avoid whipsaw
            lips_above_teeth = lips_12h[i] > teeth_12h[i]
            teeth_above_jaw = teeth_12h[i] > jaw_12h[i]
            lips_below_teeth = lips_12h[i] < teeth_12h[i]
            teeth_below_jaw = teeth_12h[i] < jaw_12h[i]
            
            # Strong trend filter: price must be outside the Alligator's mouth
            price_above_lips = close[i] > lips_12h[i]
            price_below_lips = close[i] < lips_12h[i]
            
            # Volume confirmation
            vol_ok = volume[i] > vol_ma20[i]
            
            # Volatility filter: avoid low volatility chop
            vol_filter = atr10_12h[i] > np.nanmedian(atr10_12h[max(0, i-50):i+1]) * 0.8
            
            # Long: bullish alignment + price above lips + volume + vol filter
            if (lips_above_teeth and teeth_above_jaw and price_above_lips and vol_ok and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + price below lips + volume + vol filter
            elif (lips_below_teeth and teeth_below_jaw and price_below_lips and vol_ok and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: lips crosses below teeth (trend weakening) OR price closes below jaw
            exit_cond = (lips_12h[i] < teeth_12h[i]) or (close[i] < jaw_12h[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: lips crosses above teeth OR price closes above jaw
            exit_cond = (lips_12h[i] > teeth_12h[i]) or (close[i] > jaw_12h[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator on 1d timeframe to identify strong trends, 
# entering on 12h when Alligator shows clear alignment (lips-teeth-jaw in order) 
# with price outside the mouth. Uses volume confirmation and volatility filter 
# to avoid chop. Works in bull markets by catching trends, in bear markets by 
# catching strong downtrends. Targets 50-150 trades over 4 years (12-37/year) 
# to minimize fee drag. Uses discrete sizing (0.25) to reduce churn. Effective 
# on BTC/ETH as Alligator filters out noise and captures institutional trends.