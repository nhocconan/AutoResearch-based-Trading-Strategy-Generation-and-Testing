#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# RSI(14) < 30 for long, > 70 for short only when aligned with 4h EMA(20) trend.
# Volume > 1.3x 20-bar median ensures participation.
# Session filter (08-20 UTC) reduces noise.
# Conservative sizing (0.20) to limit trades to 15-37/year target.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(20) for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: current > 1.3x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.3 * vol_median
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Skip outside session
        if not in_session[i]:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Long: RSI oversold + price above 4h EMA + volume spike
        if (rsi[i] < 30 and 
            close[i] > ema_20_4h_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.20
        
        # Short: RSI overbought + price below 4h EMA + volume spike
        elif (rsi[i] > 70 and 
              close[i] < ema_20_4h_aligned[i] and 
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