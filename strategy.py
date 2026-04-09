#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike
# - Williams %R(14) from 6h: long when < -80 (oversold), short when > -20 (overbought)
# - Trend filter: 1w EMA(34) - price must be above EMA for longs, below for shorts
# - Volume confirmation: 6h volume > 1.5x 20-period average to avoid low-vol fakeouts
# - Exits: Williams %R crosses back through -50 (mean reversion completion)
# - Position size: 0.25 (25% of capital) to limit drawdown in volatile 6h bars
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# - Williams %R is effective at capturing exhaustion moves, EMA filter avoids counter-trend trades

name = "6h_1w_williamsr_meanrev_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 6h Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or close[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion complete)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion complete)
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume and trend filter
            if (williams_r[i] <= -80 and  # Oversold
                volume_spike[i] and        # Volume confirmation
                close[i] > ema_34_1w_aligned[i]):  # Uptrend filter (price above weekly EMA)
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and   # Overbought
                  volume_spike[i] and        # Volume confirmation
                  close[i] < ema_34_1w_aligned[i]):  # Downtrend filter (price below weekly EMA)
                position = -1
                signals[i] = -0.25
    
    return signals