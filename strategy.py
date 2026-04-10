#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h EMA trend filter + volume confirmation
# - Williams %R(14) identifies overbought/oversold conditions on 6h
# - Long when %R < -80 (oversold) AND price > 12h EMA(50) (uptrend) AND volume > 1.3x 20-period average
# - Short when %R > -20 (overbought) AND price < 12h EMA(50) (downtrend) AND volume > 1.3x 20-period average
# - Exit when %R returns to -50 (mean reversion) or opposite signal
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets by combining momentum reversal with trend filter
# - Volume confirmation ensures breakouts have conviction

name = "6h_12h_williamsr_ema_volume_v1"
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
    
    # Pre-compute 6h Williams %R (14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where(highest_high == lowest_low, -50, williams_r)
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND price above 12h EMA AND volume spike
            if (williams_r[i] < -80 and  # oversold
                close[i] > ema_50_12h_aligned[i] and  # uptrend filter
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND price below 12h EMA AND volume spike
            elif (williams_r[i] > -20 and  # overbought
                  close[i] < ema_50_12h_aligned[i] and  # downtrend filter
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R returns to mean reversion level (-50) or opposite signal with volume
            exit_long = (position == 1 and 
                        (williams_r[i] >= -50 or
                         (williams_r[i] > -20 and volume_spike[i])))  # overbought exit
            exit_short = (position == -1 and 
                         (williams_r[i] <= -50 or
                          (williams_r[i] < -80 and volume_spike[i])))  # oversold exit
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals