#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d ATR regime filter
# Uses Williams Alligator (JAW=13, TEETH=8, LIPS=5) from 6h for trend direction
# 1d ATR(14) percentile filter to avoid low-volatility chop regimes
# Volume spike confirmation (2x 20-period average) for breakout validity
# Discrete position sizing 0.25 to balance risk and minimize fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Alligator provides trend-following edge in both bull and bear markets
# ATR regime filter prevents whipsaws during ranging periods
# Works on BTC/ETH by capturing sustained moves with volatility confirmation

name = "6h_WilliamsAlligator_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams Alligator from 6h
    # JAW (13-period SMMA, offset 8 bars)
    jaw_6h = pd.Series(df_6h['close'].values).rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH (8-period SMMA, offset 5 bars)
    teeth_6h = pd.Series(df_6h['close'].values).rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS (5-period SMMA, offset 3 bars)
    lips_6h = pd.Series(df_6h['close'].values).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to primary 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (50-period lookback) for regime filter
    atr_percentile = pd.Series(atr_14).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align ATR percentile to primary 6h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 6h volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and ATR percentile)
    start_idx = 50  # max(13,8,5) + offsets + ATR percentile lookback
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # ATR regime filter: only trade when volatility is above 30th percentile
        # Avoids low-volatility chop regimes where Alligator whipsaws
        if atr_percentile_aligned[i] < 30:
            volatility_filter = False
        else:
            volatility_filter = True
        
        if position == 0:  # Flat - look for new entries
            # Alligator bullish: LIPS > TEETH > JAW (green alignment)
            bullish = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
            # Alligator bearish: LIPS < TEETH < JAW (red alignment)
            bearish = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
            
            # Long: Alligator bullish AND volume confirm AND volatility filter
            if (bullish and 
                volume_confirm[i] and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND volume confirm AND volatility filter
            elif (bearish and 
                  volume_confirm[i] and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR volatility drops significantly
            bearish = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
            if bearish or (not volatility_filter and atr_percentile_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR volatility drops significantly
            bullish = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
            if bullish or (not volatility_filter and atr_percentile_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals