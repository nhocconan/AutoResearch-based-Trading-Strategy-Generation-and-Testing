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
    
    # Get 4h and 1d HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Supertrend (ATR=10, mult=3) for trend direction
    tr1 = pd.Series(df_4h['high'] - df_4h['low'])
    tr2 = pd.Series(np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]])))
    tr3 = pd.Series(np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]])))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr_4h.ewm(span=10, adjust=False, min_periods=10).mean().values
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    upper_band = hl2 + 3 * atr_10
    lower_band = hl2 - 3 * atr_10
    supertrend = np.zeros_like(df_4h['close'])
    for i in range(1, len(df_4h)):
        if df_4h['close'].iloc[i] > upper_band[i-1]:
            supertrend[i] = lower_band[i]
        elif df_4h['close'].iloc[i] < lower_band[i-1]:
            supertrend[i] = upper_band[i]
        else:
            supertrend[i] = supertrend[i-1]
            if supertrend[i] > df_4h['close'].iloc[i]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
    supertrend_dir = np.where(df_4h['close'] > supertrend, 1, -1)
    
    # 1d EMA200 for long-term trend filter
    ema_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 1h timeframe
    supertrend_1h = align_htf_to_ltf(prices, df_4h, supertrend_dir)
    ema_200_1h = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # 1h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_1h[i]) or np.isnan(ema_200_1h[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Long conditions: 4h uptrend, price above daily EMA200, RSI < 30 (oversold)
        if (supertrend_1h[i] == 1 and          # 4h uptrend
            close[i] > ema_200_1h[i] and       # Price above long-term trend
            rsi_values[i] < 30):               # Oversold entry
            signals[i] = 0.20
            
        # Short conditions: 4h downtrend, price below daily EMA200, RSI > 70 (overbought)
        elif (supertrend_1h[i] == -1 and       # 4h downtrend
              close[i] < ema_200_1h[i] and     # Price below long-term trend
              rsi_values[i] > 70):             # Overbought entry
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Supertrend_EMA200_RSI_Session"
timeframe = "1h"
leverage = 1.0