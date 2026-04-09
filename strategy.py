#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike
# - Uses Williams %R(14) on 6h for overbought/oversold signals
# - 1w EMA(34) as trend filter: only take longs above EMA, shorts below EMA
# - Volume confirmation: current volume > 2.0 * 20-period volume average
# - Works in bull markets via pullbacks to support in uptrend
# - Works in bear markets via bounces off resistance in downtrend
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to avoid fee drag
# - Williams %R provides timely mean reversion signals in ranging markets within trends

name = "6h_1w_williamsr_meanrev_trend_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute Williams %R(14) on 6h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute volume confirmation: volume > 2.0 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R exits overbought or trend change
            if williams_r[i] > -20:  # Exit overbought
                position = 0
                signals[i] = 0.0
            elif close[i] < ema_34_1w_aligned[i]:  # Exit if price closes below weekly EMA (trend change)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R exits oversold or trend change
            if williams_r[i] < -80:  # Exit oversold
                position = 0
                signals[i] = 0.0
            elif close[i] > ema_34_1w_aligned[i]:  # Exit if price closes above weekly EMA (trend change)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with volume confirmation and trend filter
            if williams_r[i] < -80 and close[i] > ema_34_1w_aligned[i] and volume_confirm[i]:  # Oversold in uptrend
                position = 1
                signals[i] = 0.25
            elif williams_r[i] > -20 and close[i] < ema_34_1w_aligned[i] and volume_confirm[i]:  # Overbought in downtrend
                position = -1
                signals[i] = -0.25
    
    return signals