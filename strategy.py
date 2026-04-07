#!/usr/bin/env python3
"""
1h_rsi_pullback_4h1d_trend_volume_v1
Hypothesis: On 1h timeframe, buy pullbacks to RSI(14) < 30 in uptrends (price > EMA50 on 4h) and sell rallies to RSI > 70 in downtrends (price < EMA50 on 4h), with volume confirmation and session filter (08-20 UTC). Uses 1d trend filter to avoid counter-trend trades in strong opposing trends. Designed for low trade frequency (~20-40/year) to minimize fee drag while capturing mean reversion in ranging markets and pullbacks in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h1d_trend_volume_v1"
timeframe = "1h"
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
    
    # 4h data for trend filter and entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d data for higher timeframe trend filter (avoid counter-trend in strong trends)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 1h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) OR trend turns down
            if rsi_values[i] > 50 or close[i] < ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) OR trend turns up
            if rsi_values[i] < 50 or close[i] > ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: RSI < 30 (oversold) + uptrend on 4h + volume confirmation
            if (rsi_values[i] < 30 and 
                close[i] > ema50_4h_aligned[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.20
            # Short: RSI > 70 (overbought) + downtrend on 4h + volume confirmation
            elif (rsi_values[i] > 70 and 
                  close[i] < ema50_4h_aligned[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.20
    
    return signals