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
    open_time = prices['open_time']
    
    # Pre-compute session filter (UTC 8-20)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    delta = pd.Series(df_4h['close'].values).diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    rsi_14_4h = (100 - (100 / (1 + rs))).values
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Calculate 4h EMA(50)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3 = np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h RSI > 50 (bullish momentum)
        # 2. Price above 4h EMA50 (bullish trend)
        # 3. Low volatility environment (ATR < 1.5% of price) to avoid choppy markets
        if (rsi_14_4h_aligned[i] > 50 and
            close[i] > ema_50_4h_aligned[i] and
            atr_14_4h_aligned[i] < 0.015 * close[i]):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h RSI < 50 (bearish momentum)
        # 2. Price below 4h EMA50 (bearish trend)
        # 3. Low volatility environment (ATR < 1.5% of price) to avoid choppy markets
        elif (rsi_14_4h_aligned[i] < 50 and
              close[i] < ema_50_4h_aligned[i] and
              atr_14_4h_aligned[i] < 0.015 * close[i]):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_RSI50_EMA50_VolFilter_Session_v1"
timeframe = "1h"
leverage = 1.0