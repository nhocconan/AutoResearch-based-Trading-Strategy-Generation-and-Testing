#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_RSI_Chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    4h KAMA trend + RSI + Chop filter. 
    - KAMA direction: long when price > KAMA, short when price < KAMA
    - RSI filter: long only when RSI > 50, short only when RSI < 50
    - Chop filter: only trade when Chop > 61.8 (ranging) for mean reversion
    - Exit: opposite signal or Chop < 38.2 (trending)
    - Uses 14-period RSI and 14-period Chop
    - Target: 20-40 trades/year on 4h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (10-period ER, 2 and 30 SC)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Chop (14)
    atr = np.zeros_like(close)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr, axis=1) / (max_high - min_low)) / np.log10(14)
    chop = chop.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in ranging market (Chop > 61.8)
            if chop[i] > 61.8:
                # Long: price above KAMA and RSI > 50
                if close[i] > kama[i] and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                # Short: price below KAMA and RSI < 50
                elif close[i] < kama[i] and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: opposite signal or trending market (Chop < 38.2)
            if close[i] < kama[i] or rsi[i] < 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: opposite signal or trending market (Chop < 38.2)
            if close[i] > kama[i] or rsi[i] > 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals