#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1h close prices for indicator calculations (already available in prices)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 6h ATR (14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h EMA (21)
    ema = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 12h close prices for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA (50) for trend direction
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(ema[i]) or 
            np.isnan(ema_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        atr_val = atr[i]
        ema_val = ema[i]
        ema_12h_val = ema_12h_aligned[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = price > ema_12h_val
        downtrend = price < ema_12h_val
        
        if position == 0:
            # Long entry: RSI oversold (<30) + price above EMA21 + uptrend
            if rsi_val < 30 and price > ema_val and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) + price below EMA21 + downtrend
            elif rsi_val > 70 and price < ema_val and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long: RSI overbought (>70) or price below EMA21
                if rsi_val > 70 or price < ema_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: RSI oversold (<30) or price above EMA21
                if rsi_val < 30 or price > ema_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_RSI_EMA_Trend_Follow_v1"
timeframe = "6h"
leverage = 1.0