#!/usr/bin/env python3
name = "12h_VolumeWeighted_RSI_TrendFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D DATA FOR TREND FILTER AND RSI ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 14-period RSI on 1d closes
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # === VOLUME WEIGHTED PRICE MOMENTUM (12-period) ===
    # Price change weighted by volume
    price_change = np.diff(close, prepend=close[0])
    vol_weighted_change = price_change * volume
    vol_weighted_ma = pd.Series(vol_weighted_change).rolling(window=12, min_periods=12).mean().values
    vol_weighted_mean = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_weighted_momentum = np.divide(
        vol_weighted_ma, 
        vol_weighted_mean, 
        out=np.zeros_like(vol_weighted_ma), 
        where=vol_weighted_mean!=0
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 12, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_12h[i]) or np.isnan(rsi_12h[i]) or 
            np.isnan(volume_weighted_momentum[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above trend + RSI not overbought + positive volume-weighted momentum
            if (close[i] > ema34_12h[i] and 
                rsi_12h[i] < 70 and
                volume_weighted_momentum[i] > 0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below trend + RSI not oversold + negative volume-weighted momentum
            elif (close[i] < ema34_12h[i] and 
                  rsi_12h[i] > 30 and
                  volume_weighted_momentum[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below trend OR RSI overbought
            if close[i] < ema34_12h[i] or rsi_12h[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above trend OR RSI oversold
            if close[i] > ema34_12h[i] or rsi_12h[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals