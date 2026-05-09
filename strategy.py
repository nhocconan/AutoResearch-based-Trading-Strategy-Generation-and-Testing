# 1h_AggressiveTrend_Scalper_v1
# Hypothesis: 1-hour aggressive trend scalper using 4h EMA crossover with volume confirmation and RSI momentum filter.
# Designed for trending markets - captures strong momentum moves while avoiding chop.
# Uses 4h for trend direction (EMA21/EMA55 crossover) and 1h for precise entry timing.
# Volume spike (>1.5x 20-period average) confirms momentum strength.
# RSI(14) > 55 for longs, < 45 for shorts to ensure momentum alignment.
# Target: 15-35 trades/year (60-140 over 4 years) to minimize fee drag.
# Works in both bull/bear markets by following the 4h trend direction.

name = "1h_AggressiveTrend_Scalper_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 55:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMAs for trend direction
    ema_21_4h = np.full_like(close_4h, np.nan)
    ema_55_4h = np.full_like(close_4h, np.nan)
    
    if len(close_4h) >= 21:
        ema_21_4h[20] = np.mean(close_4h[0:21])
        for i in range(21, len(close_4h)):
            ema_21_4h[i] = (close_4h[i] * 2 + ema_21_4h[i-1] * 19) / 21
    
    if len(close_4h) >= 55:
        ema_55_4h[54] = np.mean(close_4h[0:55])
        for i in range(55, len(close_4h)):
            ema_55_4h[i] = (close_4h[i] * 2 + ema_55_4h[i-1] * 53) / 55
    
    # Align 4h EMAs to 1h timeframe
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    ema_55_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_55_4h)
    
    # 1h RSI for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close)):
            avg_gain[i] = (gain[i] * 13 + avg_gain[i-1]) / 14
            avg_loss[i] = (loss[i] * 13 + avg_loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume spike filter
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (volume[i] * 19 + vol_ma[i-1]) / 20
    
    volume_ratio = np.divide(volume, vol_ma, out=np.full_like(volume, np.nan), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(55, 20, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_55_4h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 4h EMA21 > EMA55 (uptrend) AND RSI > 55 AND volume spike
            if (ema_21_4h_aligned[i] > ema_55_4h_aligned[i] and 
                rsi[i] > 55 and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: 4h EMA21 < EMA55 (downtrend) AND RSI < 45 AND volume spike
            elif (ema_21_4h_aligned[i] < ema_55_4h_aligned[i] and 
                  rsi[i] < 45 and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 4h EMA21 < EMA55 (trend change) OR RSI < 40 (momentum loss)
            if (ema_21_4h_aligned[i] < ema_55_4h_aligned[i] or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 4h EMA21 > EMA55 (trend change) OR RSI > 60 (momentum loss)
            if (ema_21_4h_aligned[i] > ema_55_4h_aligned[i] or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals