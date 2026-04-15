#!/usr/bin/env python3
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
    
    # Pre-compute session filter (08-20 UTC) once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) for trend filter
    delta = pd.Series(df_4h['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = 100 - (100 / (1 + rs))
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Calculate 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1h ATR(14) for volatility filter and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_4h_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 1h ATR is elevated (> 0.4% of price)
        vol_filter = atr_14[i] > 0.004 * close[i]
        
        # Long conditions:
        # 1. 4h RSI > 50 (bullish momentum)
        # 2. Price above 4h EMA20 (bullish bias)
        # 3. Volatility filter
        if (rsi_14_4h_aligned[i] > 50 and
            close[i] > ema_20_4h_aligned[i] and
            vol_filter):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h RSI < 50 (bearish momentum)
        # 2. Price below 4h EMA20 (bearish bias)
        # 3. Volatility filter
        elif (rsi_14_4h_aligned[i] < 50 and
              close[i] < ema_20_4h_aligned[i] and
              vol_filter):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_RSI40_EMA20_VolFilter_v1"
timeframe = "1h"
leverage = 1.0