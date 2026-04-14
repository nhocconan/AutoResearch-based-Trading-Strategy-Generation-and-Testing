#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1d RSI filter and volume confirmation
# Takes long when KAMA turns up (bullish), RSI < 60 (not overbought), and 1d volume > 1.5x average
# Takes short when KAMA turns down (bearish), RSI > 40 (not oversold), and 1d volume > 1.5x average
# Exits when KAMA reverses direction or volume drops below average
# Designed to capture sustained trends while avoiding overextended moves and low-volume chop
# Target: 30-80 trades per symbol over 4 years (7.5-20/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h KAMA (ER=10, smoothing=2,30)
    close_12h = df_12h['close'].values
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0)  # placeholder for correct calculation
    # Correct ER calculation
    er = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        if i < 10:
            er[i] = np.nan
        else:
            direction = abs(close_12h[i] - close_12h[i-9])
            volatility = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Calculate 1d RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for KAMA and RSI calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: 1 if rising, -1 if falling
        if i == start:
            kama_dir = 0
        else:
            kama_dir = 1 if kama_aligned[i] > kama_aligned[i-1] else -1
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: KAMA rising, RSI not overbought, volume spike
            if (kama_dir == 1 and 
                rsi_aligned[i] < 60 and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: KAMA falling, RSI not oversold, volume spike
            elif (kama_dir == -1 and 
                  rsi_aligned[i] > 40 and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA turns down or volume drops
            if kama_dir == -1 or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA turns up or volume drops
            if kama_dir == 1 or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_KAMA_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0