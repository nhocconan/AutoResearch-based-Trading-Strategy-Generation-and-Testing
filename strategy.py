#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# In trending markets: Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear).
# Filter: 1d EMA34 confirms higher timeframe trend direction.
# Entry: Williams Alligator alignment + price outside the Alligator's mouth + volume spike (>1.5x 20-period average).
# Exit: When Alligator lines re-interlace (market losing trend) or price re-enters the mouth.
# Designed for 6h timeframe targeting 15-25 trades/year. Works in bull by catching strong uptrends,
# in bear by catching strong downtrends, avoids choppy markets via Alligator's convergence/divergence.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and volume average (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume 20-period average for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Williams Alligator on 6h data: Jaw (13), Teeth (8), Lips (5)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Price outside the Alligator's mouth
            price_above_mouth = close[i] > max(lips[i], teeth[i], jaw[i])
            price_below_mouth = close[i] < min(lips[i], teeth[i], jaw[i])
            
            # Long: Bullish alignment + price above mouth + 1d uptrend + volume spike
            if (bullish_alignment and price_above_mouth and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price below mouth + 1d downtrend + volume spike
            elif (bearish_alignment and price_below_mouth and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines re-interlace OR price re-enters the mouth
            lips_teeth_cross = (lips[i] <= teeth[i] and position == 1) or (lips[i] >= teeth[i] and position == -1)
            teeth_jaw_cross = (teeth[i] <= jaw[i] and position == 1) or (teeth[i] >= jaw[i] and position == -1)
            price_reenters_mouth = (position == 1 and close[i] <= max(lips[i], teeth[i], jaw[i])) or \
                                   (position == -1 and close[i] >= min(lips[i], teeth[i], jaw[i]))
            
            if lips_teeth_cross or teeth_jaw_cross or price_reenters_mouth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0