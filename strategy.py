#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Direction_1dRSI_1dVolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend, RSI, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # temporary, will fix in loop
    # Recalculate ER and volatility properly
    er = np.zeros_like(close_1d)
    volatility_sum = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 0
            volatility_sum[i] = 0
        else:
            change_val = np.abs(close_1d[i] - close_1d[i-1])
            volatility_sum[i] = volatility_sum[i-1] + change_val
            if volatility_sum[i] > 0:
                er[i] = change_val / volatility_sum[i]
            else:
                er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period volume average for spike detection
    vol_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_20_aligned = align_htf_to_ltf(prices, df_1d, vol_20)
    
    # Calculate 12h RSI for entry timing
    delta_12h = np.diff(close, prepend=close[0])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    avg_gain_12h = pd.Series(gain_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_12h = pd.Series(loss_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_12h = avg_gain_12h / (avg_loss_12h + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need 20 for volume, 14 for RSI
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_20_aligned[i]) or np.isnan(rsi_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        rsi_1d = rsi_aligned[i]
        vol_20_val = vol_20_aligned[i]
        rsi_12h_val = rsi_12h[i]
        vol = volume[i]
        
        if position == 0:
            # Enter long: price > KAMA (uptrend) AND RSI_1d < 30 (oversold) AND volume > 1.5x average AND RSI_12h > 40 (momentum)
            if close[i] > kama_val and rsi_1d < 30 and vol > 1.5 * vol_20_val and rsi_12h_val > 40:
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA (downtrend) AND RSI_1d > 70 (overbought) AND volume > 1.5x average AND RSI_12h < 60
            elif close[i] < kama_val and rsi_1d > 70 and vol > 1.5 * vol_20_val and rsi_12h_val < 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA OR RSI_1d > 70 (overbought)
            if close[i] < kama_val or rsi_1d > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA OR RSI_1d < 30 (oversold)
            if close[i] > kama_val or rsi_1d < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals