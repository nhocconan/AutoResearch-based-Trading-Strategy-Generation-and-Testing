#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly volume-weighted average price (VWAP) deviation with 1d trend filter.
# Goes long when price deviates significantly below weekly VWAP in an uptrend, short when above in downtrend.
# Uses weekly structure for mean reversion zones and daily trend to avoid counter-trend trades.
# Designed for low trade frequency (15-30/year) to avoid fee drag in ranging/bear markets.

name = "6h_WeeklyVWAP_Deviation_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    pv_1w = typical_price_1w * volume_1w
    
    # Cumulative sums for VWAP
    cum_pv = np.cumsum(pv_1w)
    cum_vol = np.cumsum(volume_1w)
    
    # Avoid division by zero
    vwap_1w = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Align weekly VWAP to 6h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Get daily data for trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate deviation from weekly VWAP as percentage
        vwap_dev_pct = (close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i] * 100
        
        if position == 0:
            # Enter long: price significantly below weekly VWAP AND daily uptrend (price > EMA34)
            if vwap_dev_pct < -1.5 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price significantly above weekly VWAP AND daily downtrend (price < EMA34)
            elif vwap_dev_pct > 1.5 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly VWAP or trend breaks
            if vwap_dev_pct > -0.5 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly VWAP or trend breaks
            if vwap_dev_pct < 0.5 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals