#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI mean reversion with 4-hour EMA filter and volume confirmation
# RSI(14) < 30 for long entry, > 70 for short entry, only in direction of 4h EMA(20)
# Volume must be > 1.5x 20-bar median to ensure participation
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# Conservative sizing (0.20) to limit trade frequency and avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA(20) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Long: RSI < 30 + price above 4h EMA + volume spike
        if (rsi[i] < 30 and 
            close[i] > ema_4h_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.20
        
        # Short: RSI > 70 + price below 4h EMA + volume spike
        elif (rsi[i] > 70 and 
              close[i] < ema_4h_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.20
        
        # Exit: RSI returns to neutral zone (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.20 and rsi[i] > 40) or
               (signals[i-1] == -0.20 and rsi[i] < 60))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_RSI_MeanReversion_4hEMAFilter_Volume"
timeframe = "1h"
leverage = 1.0