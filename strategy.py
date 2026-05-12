#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_4hTrendFilter_v1
Hypothesis: In 1h timeframe, RSI mean reversion combined with 4h trend filter (EMA50) and session filter (08-20 UTC) captures short-term reversals within the dominant trend, reducing false signals. Works in bull markets via pullbacks and in bear markets via bounces. Volume confirmation ensures legitimacy. Target: 20-40 trades/year per symbol.
"""

name = "1h_RSI_MeanReversion_4hTrendFilter_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0).values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC (pre-compute hour from index)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # LONG: RSI < 30 (oversold) + price above 4h EMA50 (uptrend) + volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: RSI > 70 (overbought) + price below 4h EMA50 (downtrend) + volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (mean reversion complete) OR price below 4h EMA50 (trend change)
            if (rsi[i] > 50) or (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI < 50 (mean reversion complete) OR price above 4h EMA50 (trend change)
            if (rsi[i] < 50) or (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals