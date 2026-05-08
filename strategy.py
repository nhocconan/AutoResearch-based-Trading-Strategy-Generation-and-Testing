#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA Trend + RSI + Volume Spike
# - Uses KAMA direction (bullish/bearish) from 4h as primary trend filter
# - RSI(14) for momentum confirmation (long when RSI>55, short when RSI<45)
# - Volume spike (>2x 20-period average) confirms breakout strength
# - Works in bull/bear by using KAMA which adapts to market noise
# - Target: 20-30 trades/year to minimize fee drag on 4h timeframe

name = "4h_KAMA_RSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change.rolling(window=10, min_periods=10).sum() / volatility.replace(0, np.finfo(float).eps)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = kama > np.roll(kama, 1)
    kama_dir[0] = False
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA bullish + RSI>55 + volume spike
            long_cond = kama_dir[i] and (rsi[i] > 55) and volume_spike[i]
            
            # Short: KAMA bearish + RSI<45 + volume spike
            short_cond = (~kama_dir[i]) and (rsi[i] < 45) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns bearish OR RSI<45
            if (~kama_dir[i]) or (rsi[i] < 45):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns bullish OR RSI>55
            if kama_dir[i] or (rsi[i] > 55):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals