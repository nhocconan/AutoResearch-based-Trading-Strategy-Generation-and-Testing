#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h EMA34 trend filter and 1h VWAP deviation for mean reversion entries.
# Uses 4h EMA34 for trend direction (avoid counter-trend trades) and enters when price deviates >1.5σ from 1h VWAP.
# Only trades during 08-20 UTC session to avoid low-volume noise.
# Targets 15-37 trades/year (60-150 total over 4 years) by requiring both trend alignment and significant deviation.
# Works in bull/bear by following higher timeframe trend while exploiting short-term mean reversion.
name = "1h_4h_EMA34_VWAPDev_MeanRev"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA34 trend (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h VWAP and standard deviation for mean reversion signals
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_denominator = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = vwap_numerator / vwap_denominator
    
    # Calculate standard deviation of price deviation from VWAP
    price_dev = typical_price - vwap
    dev_std = pd.Series(price_dev).rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for VWAP calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(dev_std[i]) or dev_std[i] == 0 or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Calculate z-score of price deviation from VWAP
        z_score = price_dev[i] / dev_std[i]
        
        if position == 0:
            # Long: uptrend (price > 4h EMA34) AND price significantly below VWAP (mean reversion long)
            if (close[i] > ema_34_4h_aligned[i] and 
                z_score < -1.5):
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price < 4h EMA34) AND price significantly above VWAP (mean reversion short)
            elif (close[i] < ema_34_4h_aligned[i] and 
                  z_score > 1.5):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if trend breaks or price reverts to VWAP
            if close[i] < ema_34_4h_aligned[i] or z_score > -0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if trend breaks or price reverts to VWAP
            if close[i] > ema_34_4h_aligned[i] or z_score < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals