#!/usr/bin/env python3
name = "1d_1w_KAMA_Direction_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily KAMA for trend direction
    delta = pd.Series(close).diff().abs()
    direction = pd.Series(close).diff().abs()
    change = abs(pd.Series(close).diff(10))
    volatility = delta.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [np.nan] * len(close)
    for i in range(1, len(close)):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    
    # Daily RSI for overbought/oversold
    delta_rsi = pd.Series(close).diff()
    gain = delta_rsi.where(delta_rsi > 0, 0)
    loss = -delta_rsi.where(delta_rsi < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Daily volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if weekly trend or volume data not ready
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: KAMA up, RSI < 30 (oversold), volume confirmation
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                volume[i] > vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA down, RSI > 70 (overbought), volume confirmation
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  volume[i] > vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when KAMA turns down or RSI > 70
            if (close[i] < kama[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when KAMA turns up or RSI < 30
            if (close[i] > kama[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals