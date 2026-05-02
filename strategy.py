#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and ATR regime filter
# Uses Williams Alligator (jaw/teeth/lips) from 1d for trend direction, 
# 1d volume spike confirms participation, 
# 1d ATR-based chop filter avoids whipsaws in ranging markets
# Discrete position sizing 0.25 minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by only taking Alligator-aligned breaks with volume
# ATR regime filter (CHOP-like) avoids false signals in low volatility

name = "12h_WilliamsAlligator_1dATR_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Williams Alligator, ATR, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator: SMA of median price
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    median_price = (df_1d['high'] + df_1d['low']) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Calculate 1d ATR(14) for volatility/chop regime filter
    tr1 = df_1d['high'][1:].values - df_1d['low'][1:].values
    tr2 = np.abs(df_1d['high'][1:].values - df_1d['close'][:-1].values)
    tr3 = np.abs(df_1d['low'][1:].values - df_1d['close'][:-1].values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume confirmation (2x 20-period average)
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = df_1d['volume'].values > (vol_ma * 2.0)
    
    # Align Alligator lines, ATR, and volume to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and ATR)
    start_idx = 35  # max(13 for jaw, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ATR regime filter: avoid low volatility (choppy) markets
        # Calculate ATR ratio to its 20-period mean to detect expansion/contraction
        atr_ma = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().shift(1).values
        if np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            volatility_filter = True  # allow trade if MA not ready
        else:
            # Trade when ATR is above 70% of its MA (avoid very low volatility chop)
            volatility_filter = atr_14_aligned[i] > (atr_ma[i] * 0.7)
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
            # Long: bullish alignment AND price breaks above lips AND volume confirm AND volatility filter
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                close[i] > lips_aligned[i] and
                volume_confirm_aligned[i] > 0.5 and  # aligned as float, treat as bool
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND price breaks below lips AND volume confirm AND volatility filter
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                  close[i] < lips_aligned[i] and
                  volume_confirm_aligned[i] > 0.5 and
                  volatility_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below teeth OR Alligator alignment turns bearish OR volatility drops
            if (close[i] < teeth_aligned[i] or
                lips_aligned[i] < teeth_aligned[i] or  # alignment broken
                (not volatility_filter and atr_14_aligned[i] < (atr_ma[i] * 0.5))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above teeth OR Alligator alignment turns bullish OR volatility drops
            if (close[i] > teeth_aligned[i] or
                lips_aligned[i] > teeth_aligned[i] or  # alignment broken
                (not volatility_filter and atr_14_aligned[i] < (atr_ma[i] * 0.5))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals