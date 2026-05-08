#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend + RSI mean reversion + volume spike
# Uses adaptive trend filter (KAMA) to avoid whipsaws in ranging markets
# Long: KAMA bullish AND RSI < 35 AND volume spike (>1.5x 20-period avg)
# Short: KAMA bearish AND RSI > 65 AND volume spike (>1.5x 20-period avg)
# Exit: RSI returns to neutral (40-60) or trend change
# Designed for 20-35 trades/year with proper risk control via mean reversion in trend

name = "4h_KAMA_RSI_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive trend filter)
    # Efficiency Ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else np.array([0.0])
    # Fix volatility calculation for rolling window
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+10]))) if i+10 <= len(close) else 0.0 
                          for i in range(len(close))])
    er = np.divide(change, volatility, out=np.full_like(change, 0.0), where=volatility!=0)
    # Smoothing constants: fastest SC=2/(2+1)=0.67, slowest SC=2/(30+1)=0.0645
    sc = np.power(er * (0.67 - 0.0645) + 0.0645, 2)
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for RSI and volatility
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for mean reversion entries within trend
            # Long: KAMA bullish (price > KAMA) AND RSI oversold AND volume spike
            if close[i] > kama[i] and rsi[i] < 35 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish (price < KAMA) AND RSI overbought AND volume spike
            elif close[i] < kama[i] and rsi[i] > 65 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral or trend change
            if rsi[i] > 40 or close[i] <= kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral or trend change
            if rsi[i] < 60 or close[i] >= kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals