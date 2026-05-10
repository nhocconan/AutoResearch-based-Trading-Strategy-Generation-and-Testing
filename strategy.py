# 1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) to capture the trend direction on daily timeframe, combined with RSI for momentum and Choppiness Index for regime filtering. Enters long when KAMA turns up, RSI is above 50, and market is trending (CHOP < 38.2). Enters short when KAMA turns down, RSI is below 50, and market is trending. Exits when conditions reverse. Designed to work in both bull and bear markets by only trading in trending regimes and avoiding sideways markets.
# Targets 10-25 trades per year on 1d timeframe with position size 0.25.

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_window = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_window))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle the volatility calculation properly for rolling window
    volatility = pd.Series(np.abs(np.diff(close))).rolling(window=er_window, min_periods=1).sum().values
    volatility = np.concatenate([np.full(er_window-1, np.nan), volatility[er_window-1:]])
    er = np.where(volatility != 0, change / volatility, 0)
    er = np.concatenate([np.full(er_window-1, np.nan), er[er_window-1:]])
    
    # Calculate smoothing constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_window-1] = close[er_window-1]  # Start with first available close
    for i in range(er_window, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])
    
    # Choppiness Index (CHOP) - 14 period
    atr = np.zeros_like(close)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(np.sum(atr, axis=1) / (max_high - min_low)) / np.log10(14), 
                    50)
    # Fix the chop calculation for rolling sum
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14), 
                    50)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])
    
    # Determine trend direction from KAMA slope
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    kama_up[0] = False
    kama_down[0] = False
    
    # Momentum filter: RSI > 50 for long, RSI < 50 for short
    rsi_above_50 = rsi > 50
    rsi_below_50 = rsi < 50
    
    # Regime filter: CHOP < 38.2 indicates trending market
    trending = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 14)  # Warmup for KAMA, RSI, and CHOP
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: KAMA turning up, RSI > 50, trending market
            if kama_up[i] and rsi_above_50[i] and trending[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA turning down, RSI < 50, trending market
            elif kama_down[i] and rsi_below_50[i] and trending[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turning down or RSI < 50 or market becomes choppy
            if kama_down[i] or not rsi_above_50[i] or not trending[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turning up or RSI > 50 or market becomes choppy
            if kama_up[i] or not rsi_below_50[i] or not trending[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals