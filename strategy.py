#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike confirmation
# - Uses 6h Williams %R(14) for overbought/oversold signals (%R < -80 long, %R > -20 short)
# - Filters with 12h EMA(50) trend: only long when price > EMA50, short when price < EMA50
# - Confirms with 6h volume > 2.0x 20-period average for institutional participation
# - Exits when Williams %R reverts to mean (-50) or opposite extreme (%R > -20 for longs, %R < -80 for shorts)
# - Position size: 0.25 (25% of capital) for controlled risk
# - Target: 12-25 trades/year on 6h timeframe (48-100 total over 4 years) to minimize fee drag
# - Works in bull markets (mean reversion in uptrend) and bear markets (mean reversion in downtrend)
# - Williams %R identifies exhaustion points; trend filter prevents fighting strong moves

name = "6h_12h_williamsr_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 6h Volume > 2.0x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or ema_50_12h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R reverts to mean (-50) or becomes overbought (> -20)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R reverts to mean (-50) or becomes oversold (< -80)
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with volume confirmation and trend filter
            if (williams_r[i] <= -80 and  # Oversold
                volume_spike[i] and       # Volume confirmation
                close[i] > ema_50_12h_aligned[i]):  # Uptrend filter
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and   # Overbought
                  volume_spike[i] and      # Volume confirmation
                  close[i] < ema_50_12h_aligned[i]):  # Downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals