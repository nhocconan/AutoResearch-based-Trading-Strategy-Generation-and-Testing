#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike
# - Uses Williams %R(14) on 6h for oversold/overbought entries (long < -80, short > -20)
# - Trend filter: 12h EMA(50) slope - only long when EMA rising, short when falling
# - Volume confirmation: volume > 2.0 * 20-period average to avoid false signals
# - Works in bull markets via pullbacks to rising EMA, in bear via bounces off falling EMA
# - Target: 12-25 trades/year on 6h timeframe (48-100 total over 4 years) to avoid fee drag
# - Williams %R is effective for mean reversion in ranging markets which dominate 2025+

name = "6h_12h_williamsr_meanrev_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope_12h = np.diff(ema_50_12h, prepend=ema_50_12h[0])  # slope = current - previous
    ema_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_12h)
    
    # Williams %R(14) on 6h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume confirmation: volume > 2.0 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_slope_12h_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R mean reversion or trend change
            if williams_r[i] > -20:  # Overbought exit
                position = 0
                signals[i] = 0.0
            elif ema_slope_12h_aligned[i] <= 0:  # Trend turned down
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R mean reversion or trend change
            if williams_r[i] < -80:  # Oversold exit
                position = 0
                signals[i] = 0.0
            elif ema_slope_12h_aligned[i] >= 0:  # Trend turned up
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with volume confirmation and trend alignment
            if williams_r[i] < -80 and ema_slope_12h_aligned[i] > 0 and volume_confirm[i]:  # Oversold + rising trend
                position = 1
                signals[i] = 0.25
            elif williams_r[i] > -20 and ema_slope_12h_aligned[i] < 0 and volume_confirm[i]:  # Overbought + falling trend
                position = -1
                signals[i] = -0.25
    
    return signals