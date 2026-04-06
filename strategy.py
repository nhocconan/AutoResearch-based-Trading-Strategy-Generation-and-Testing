#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h EMA trend filter and volume confirmation
# Enter long when: Williams %R < -80, price > 12h EMA(20), volume > 1.5x avg
# Enter short when: Williams %R > -20, price < 12h EMA(20), volume > 1.5x avg
# Exit when Williams %R returns to neutral range (-50 to -30) or opposite extreme
# Williams %R identifies overbought/oversold levels; EMA filter ensures trend alignment
# Target: 75-200 trades over 4 years (19-50/year) for optimal fee efficiency

name = "4h_williamsr_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.fillna(-50).values  # neutral when undefined
    
    # 12h EMA(20) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_20 = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Williams %R > -30 (return from oversold) OR < -80 (deep oversold) OR trend change
            if williams_r[i] > -30 or williams_r[i] < -80 or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R < -70 (return from overbought) OR > -20 (deep overbought) OR trend change
            if williams_r[i] < -70 or williams_r[i] > -20 or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extreme + trend filter + volume
            if williams_r[i] < -80 and close[i] > ema_20_aligned[i] and volume[i] > volume_threshold[i]:
                # Oversold with uptrend - bullish mean reversion
                signals[i] = 0.25
                position = 1
            elif williams_r[i] > -20 and close[i] < ema_20_aligned[i] and volume[i] > volume_threshold[i]:
                # Overbought with downtrend - bearish mean reversion
                signals[i] = -0.25
                position = -1
    
    return signals