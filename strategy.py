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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily KAMA (adaptive moving average)
    # Efficiency Ratio (ER) = |close - close(10)| / sum(|close - close(1)|, 10)
    change = np.abs(daily_close - np.roll(daily_close, 10))
    change[0:10] = np.nan  # First 10 values invalid
    
    volatility = np.zeros_like(daily_close)
    for i in range(1, len(daily_close)):
        volatility[i] = volatility[i-1] + np.abs(daily_close[i] - daily_close[i-1])
    
    # Calculate ER with proper handling
    er = np.full_like(daily_close, np.nan, dtype=float)
    for i in range(10, len(daily_close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(daily_close, np.nan, dtype=float)
    kama[0] = daily_close[0]
    for i in range(1, len(daily_close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (daily_close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate daily RSI(14)
    delta = np.diff(daily_close)
    delta = np.insert(delta, 0, 0)  # Same length as close
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value invalid
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1d timeframe (no additional delay needed for KAMA/RSI/ATR)
    kama_1d = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d = align_htf_to_ltf(prices, df_1d, rsi)
    atr_14_1d = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(atr_14_1d[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Price above KAMA + RSI < 30 (oversold) → long
        # 2. Price below KAMA + RSI > 70 (overbought) → short
        # 3. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        # 4. Discrete position sizing: 0.25
        
        # Long conditions: price above KAMA and RSI oversold
        if (close[i] > kama_1d[i] and            # Price above KAMA (bullish bias)
            rsi_1d[i] < 30 and                   # RSI oversold
            atr_14_1d[i] > 0.003 * close[i]):    # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: price below KAMA and RSI overbought
        elif (close[i] < kama_1d[i] and          # Price below KAMA (bearish bias)
              rsi_1d[i] > 70 and                 # RSI overbought
              atr_14_1d[i] > 0.003 * close[i]):  # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Volatility_Filter"
timeframe = "1d"
leverage = 1.0