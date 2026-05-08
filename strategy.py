#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action relative to 12h VWAP with volume confirmation and 1d EMA trend filter.
# Long when price crosses above 12h VWAP AND 4h volume > 1.5x 20-period average AND price > 1d EMA50.
# Short when price crosses below 12h VWAP AND 4h volume > 1.5x 20-period average AND price < 1d EMA50.
# Exit when price crosses back below/above 1d EMA50 (trend-based exit).
# Uses VWAP as dynamic support/resistance and volume spike for institutional interest confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drag.

name = "4h_VWAP_Cross_12h_Volume_1dEMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h VWAP: cumulative(VP) / cumulative(volume)
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    volume_price_12h = typical_price_12h * df_12h['volume']
    cum_vp_12h = volume_price_12h.cumsum().values
    cum_vol_12h = df_12h['volume'].cumsum().values
    vwap_12h = np.divide(cum_vp_12h, cum_vol_12h, out=np.full_like(cum_vp_12h, np.nan), where=cum_vol_12h!=0)
    
    # Align 12h VWAP to 4h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_12h_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price crosses above VWAP, volume spike, above 1d EMA50
            long_cond = (close[i] > vwap_12h_aligned[i]) and volume_filter[i] and (close[i] > ema50_1d_aligned[i])
            # Short conditions: price crosses below VWAP, volume spike, below 1d EMA50
            short_cond = (close[i] < vwap_12h_aligned[i]) and volume_filter[i] and (close[i] < ema50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d EMA50 (trend change)
            if close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1d EMA50 (trend change)
            if close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals