#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w EMA trend filter + volume confirmation
# - Uses 1w EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - Uses 12h Williams Alligator (Jaw=TEETH=LIPS) for trend strength and entry signals
# - Requires volume > 1.3 * 20-period volume average for confirmation
# - Fixed position size 0.25 to manage drawdown and reduce fee churn
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)
# - Williams Alligator catches strong trends in both bull and bear markets
# - 1w EMA filter ensures we only trade with the higher timeframe trend
# - Volume confirmation reduces false breakouts

name = "12h_1w_alligator_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Pre-compute 12h Williams Alligator
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    median_price = (high + low) / 2.0
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().values  # Smoothed by 8
    
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().values  # Smoothed by 5
    
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().values  # Smoothed by 3
    
    # Align Alligator components to 12h timeframe (already on 12h, no alignment needed)
    # But we need to ensure proper warmup
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Pre-compute volume confirmation: volume > 1.3 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Williams Alligator signals:
        # - Alligator sleeping (JAW ≈ TEETH ≈ LIPS): no trend, avoid trading
        # - Alligator waking up (JAW, TEETH, LIPS diverging): trend forming
        # - Alligator eating (JAW > TEETH > LIPS for uptrend, JAW < TEETH < LIPS for downtrend): strong trend
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Alligator alignment for uptrend: JAW > TEETH > LIPS
        alligator_uptrend = jaw_val > teeth_val and teeth_val > lips_val
        # Alligator alignment for downtrend: JAW < TEETH < LIPS
        alligator_downtrend = jaw_val < teeth_val and teeth_val < lips_val
        
        # Alligator sleeping (all lines intertwined): avoid trading
        jaw_teeth_diff = abs(jaw_val - teeth_val)
        teeth_lips_diff = abs(teeth_val - lips_val)
        jaws_lips_diff = abs(jaw_val - lips_val)
        alligator_sleeping = (jaw_teeth_diff < 0.001 * close[i] and 
                             teeth_lips_diff < 0.001 * close[i] and
                             jaws_lips_diff < 0.001 * close[i])
        
        if position == 1:  # Long position
            # Exit conditions: trend weakness or reversal
            if not alligator_uptrend or alligator_sleeping or close[i] < lips_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: trend weakness or reversal
            if not alligator_downtrend or alligator_sleeping or close[i] > lips_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries in direction of 1w trend with Alligator confirmation and volume
            if uptrend and alligator_uptrend and not alligator_sleeping and volume_confirm[i]:
                position = 1
                signals[i] = 0.25
            elif downtrend and alligator_downtrend and not alligator_sleeping and volume_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals