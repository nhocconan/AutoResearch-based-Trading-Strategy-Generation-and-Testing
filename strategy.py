#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action near 12h VWAP with 1d trend filter and volume confirmation.
# Long when price crosses above 12h VWAP AND 1d EMA50 > EMA200 (uptrend) AND 4h volume > 1.5x 20-period average.
# Short when price crosses below 12h VWAP AND 1d EMA50 < EMA200 (downtrend) AND 4h volume > 1.5x 20-period average.
# Exit when price crosses back below/above 12h VWAP.
# Uses VWAP for mean reversion in intraday trends with higher timeframe trend filter.
# Target: 100-200 total trades over 4 years (25-50/year) for balanced frequency.

name = "4h_VWAP_1dTrend_Volume"
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
    
    # 4h VWAP calculation (typical price * volume / cumulative volume)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_trend = ema50 - ema200  # Positive for uptrend, negative for downtrend
    
    # Align 1d EMA trend to 4h timeframe
    ema_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price crosses above VWAP, uptrend, volume spike
            long_cond = (close[i] > vwap[i]) and (close[i-1] <= vwap[i-1]) and (ema_trend_aligned[i] > 0) and volume_filter[i]
            # Short conditions: price crosses below VWAP, downtrend, volume spike
            short_cond = (close[i] < vwap[i]) and (close[i-1] >= vwap[i-1]) and (ema_trend_aligned[i] < 0) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP
            if close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above VWAP
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals