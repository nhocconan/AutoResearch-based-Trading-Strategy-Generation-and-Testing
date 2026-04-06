#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-day trend filter and volume confirmation
# Enter long when: Williams %R < -80 (oversold), price > 1d EMA(50), volume > 1.5x avg, during active session (08-20 UTC)
# Enter short when: Williams %R > -20 (overbought), price < 1d EMA(50), volume > 1.5x avg, during active session
# Exit when Williams %R returns to neutral zone (-50 to -30) or opposite extreme reached
# Williams %R is more responsive than RSI for mean reversion, targeting 50-150 trades over 4 years

name = "6h_williamsr_1dema_vol_session_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    willr = willr.values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(willr[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: Williams %R > -30 OR Williams %R < -80 (deep oversold) OR price < 1d EMA(50)
            if willr[i] > -30 or willr[i] < -80 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R < -70 OR Williams %R > -20 (deep overbought) OR price > 1d EMA(50)
            if willr[i] < -70 or willr[i] > -20 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extreme + trend filter + volume + session
            if in_session and volume[i] > volume_threshold[i]:
                if willr[i] < -80 and close[i] > ema_50_aligned[i]:
                    # Oversold but above daily EMA - bullish mean reversion
                    signals[i] = 0.25
                    position = 1
                elif willr[i] > -20 and close[i] < ema_50_aligned[i]:
                    # Overbought but below daily EMA - bearish mean reversion
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>