# 4h_TRIX_VolumeSpike_12hTrend - TRIX momentum with volume confirmation and 12h trend filter
# TRIX filters noise and detects momentum shifts. Volume spike confirms breakout strength.
# 12h EMA trend filter ensures alignment with higher timeframe direction.
# Target: 20-40 trades/year to avoid fee drag, works in bull/bear via trend filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15,9,9) on 4h close
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9) - 1 period rate of change
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = ema3.pct_change() * 100  # Convert to percentage
    trix_values = trix.values
    
    # 12h EMA50 for trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume spike detection (4h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Wait for TRIX to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_values[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]  # Volume spike confirmation
        
        if position == 0:
            # Long: TRIX turning up + above zero + 12h uptrend + volume spike
            if (trix_values[i] > trix_values[i-1] and  # TRIX rising
                trix_values[i] > 0 and                 # Above zero line
                close[i] > ema_12h_aligned[i] and      # Above 12h EMA (uptrend)
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: TRIX turning down + below zero + 12h downtrend + volume spike
            elif (trix_values[i] < trix_values[i-1] and  # TRIX falling
                  trix_values[i] < 0 and                 # Below zero line
                  close[i] < ema_12h_aligned[i] and      # Below 12h EMA (downtrend)
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns down or trend turns down
            if trix_values[i] < trix_values[i-1] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns up or trend turns up
            if trix_values[i] > trix_values[i-1] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals