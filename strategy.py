#!/usr/bin/env python3
"""
4h_rsi_ema_pullback_12h_trend_volume_v1
Hypothesis: On 4h timeframe, buy pullbacks to EMA20 during 12h uptrends with RSI < 40 and volume confirmation.
Sell/short when RSI > 60 or price breaks below EMA20 in uptrend, or vice versa for downtrends.
Uses 12h EMA50 as trend filter to avoid counter-trend trades. Designed for ~25-40 trades/year.
Works in bull markets via trend-following pullbacks and in bear markets via counter-trend bounces at EMA20.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_ema_pullback_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_12h = df_12h['close'].ewm(span=50, adjust=False).mean()
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h.values)
    
    # EMA20 on 4h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # RSI(14) on 4h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation (20-period average on 4h = ~3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x average volume
        vol_confirm = volume[i] > 1.2 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 60 (overbought) or price closes below EMA20
            if rsi[i] > 60 or close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: RSI < 40 (oversold) or price closes above EMA20
            if rsi[i] < 40 or close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: pullback to EMA20 in 12h uptrend with RSI < 40 and volume
            if (close[i] >= ema_20[i] * 0.995 and close[i] <= ema_20[i] * 1.005 and  # near EMA20
                ema_12h_aligned[i] > ema_20[i] and  # 12h uptrend filter (price > EMA)
                rsi[i] < 40 and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: pullback to EMA20 in 12h downtrend with RSI > 60 and volume
            elif (close[i] >= ema_20[i] * 0.995 and close[i] <= ema_20[i] * 1.005 and  # near EMA20
                  ema_12h_aligned[i] < ema_20[i] and  # 12h downtrend filter (price < EMA)
                  rsi[i] > 60 and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals