#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; 1d EMA filters for trend alignment
# Volume confirms momentum; avoids choppy markets
# Targets 50-150 trades over 4 years by using strict entry conditions
# Works in bull/bear markets: trend filter prevents counter-trend trades in strong moves

name = "6h_williamsr_1dema_vol_v1"
timeframe = "6h"
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
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when undefined
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Wait for Williams %R to stabilize
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Williams %R > -20 (overbought) OR price < 1d EMA(50) (trend break)
            if williams_r[i] > -20 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R < -80 (oversold) OR price > 1d EMA(50) (trend break)
            if williams_r[i] < -80 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extreme + trend filter + volume
            if williams_r[i] < -80 and close[i] > ema_50_aligned[i]:
                # Oversold but above daily EMA - bullish mean reversion in uptrend
                signals[i] = 0.25
                position = 1
            elif williams_r[i] > -20 and close[i] < ema_50_aligned[i]:
                # Overbought but below daily EMA - bearish mean reversion in downtrend
                signals[i] = -0.25
                position = -1
    
    return signals