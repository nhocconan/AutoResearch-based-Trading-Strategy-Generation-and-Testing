# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_RSI2_Regime_Breakout
RSI(2) mean reversion with trend filter and volatility regime:
- Long when RSI(2) < 10 and price > EMA(200) and ATR(14) < SMA(ATR(14), 50)
- Short when RSI(2) > 90 and price < EMA(200) and ATR(14) < SMA(ATR(14), 50)
- Exit when RSI(2) crosses 50 (mean reversion complete)
- Uses 12h EMA(50) for higher timeframe trend confirmation
- Designed for 20-40 trades/year per symbol
Works in both bull (buying dips in uptrend) and bear (selling rallies in downtrend) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA(200) for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) for volatility regime
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate SMA of ATR(50) for volatility regime filter
    atr_sma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need sufficient data for EMA(200)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_200[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_sma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: low volatility (ATR < SMA of ATR)
        low_volatility = atr[i] < atr_sma[i]
        
        # Trend filter: price vs EMA(200) and 12h EMA(50)
        uptrend = close[i] > ema_200[i] and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
        downtrend = close[i] < ema_200[i] and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
        
        if position == 0:
            # Long: RSI(2) oversold + uptrend + low volatility
            if rsi[i] < 10 and uptrend and low_volatility:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) overbought + downtrend + low volatility
            elif rsi[i] > 90 and downtrend and low_volatility:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) crosses above 50 (mean reversion complete)
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI(2) crosses below 50 (mean reversion complete)
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI2_Regime_Breakout"
timeframe = "4h"
leverage = 1.0