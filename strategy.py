#!/usr/bin/env python3
# 12h_1d_vwap_mean_reversion_v2
# Strategy: 12h VWAP mean reversion with 1d trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price reverts to VWAP in ranging markets. In trending markets, we follow the trend. Uses 1d EMA50 for trend filter and volume > 1.5x average for confirmation. Targets 20-50 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_vwap_mean_reversion_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # VWAP calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # Distance from VWAP as percentage
    vwap_dist = (close - vwap) / vwap
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vwap_dist[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: VWAP mean reversion in ranging, trend following in trending
        if (vwap_dist[i] < -0.008 and vol_confirm[i] and not uptrend and position != 1):  # Oversold in downtrend/ranging
            position = 1
            signals[i] = 0.25
        elif (vwap_dist[i] > 0.008 and vol_confirm[i] and not downtrend and position != -1):  # Overbought in uptrend/ranging
            position = -1
            signals[i] = -0.25
        # Exit: VWAP cross or trend change
        elif position == 1 and (vwap_dist[i] > 0.002 or downtrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (vwap_dist[i] < -0.002 or uptrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals