#!/usr/bin/env python3
"""
1h_RSI45_Trend_Volume_Spike
Hypothesis: RSI crosses above 45 in strong uptrend (price > EMA200) with volume spikes signal momentum continuation. Crosses below 45 in downtrend (price < EMA200) with volume spikes signal reversals. Uses 1h timeframe with 4h trend filter and volume confirmation to limit trades to 15-30/year. Works in bull via RSI >45 uptrend continuation and in bear via RSI <45 downtrend continuation.
"""

name = "1h_RSI45_Trend_Volume_Spike"
timeframe = "1h"
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
    
    # Get 4h data for EMA200 trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA200 for trend filter
    ema200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if position == 0:
            # LONG: RSI crosses above 45 with volume spike and uptrend
            if (rsi_values[i] > 45 and rsi_values[i-1] <= 45 and 
                volume_spike[i] and 
                close[i] > ema200_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: RSI crosses below 45 with volume spike and downtrend
            elif (rsi_values[i] < 45 and rsi_values[i-1] >= 45 and 
                  volume_spike[i] and 
                  close[i] < ema200_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI falls below 40 or trend reverses
            if (rsi_values[i] < 40) or \
               (close[i] < ema200_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI rises above 60 or trend reverses
            if (rsi_values[i] > 60) or \
               (close[i] > ema200_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals