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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 4h RSI(14) for momentum filter
    delta = pd.Series(df_4h['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = 100 - (100 / (1 + rs))
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Get 1d HTF data for session-independent volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # track current position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(rsi_14_4h_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        if not (in_session and vol_filter):
            signals[i] = 0.0
            position = 0
            continue
        
        # Long conditions:
        # 1. Price above 4h EMA20 (bullish bias)
        # 2. 4h RSI between 30 and 50 (bullish momentum, not overbought)
        # 3. Close above open (bullish 1h candle)
        if (close[i] > ema_20_4h_aligned[i] and
            30 <= rsi_14_4h_aligned[i] <= 50 and
            close[i] > prices['open'].iloc[i]):
            signals[i] = 0.20
            position = 1
            
        # Short conditions:
        # 1. Price below 4h EMA20 (bearish bias)
        # 2. 4h RSI between 50 and 70 (bearish momentum, not oversold)
        # 3. Close below open (bearish 1h candle)
        elif (close[i] < ema_20_4h_aligned[i] and
              50 <= rsi_14_4h_aligned[i] <= 70 and
              close[i] < prices['open'].iloc[i]):
            signals[i] = -0.20
            position = -1
        else:
            # Exit conditions: flatten position when conditions fail
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1h_EMA20_RSI14_SessionVolFilter_v1"
timeframe = "1h"
leverage = 1.0