#!/usr/bin/env python3
name = "4h_RSI_MeanReversion_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 4h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion: RSI oversold in uptrend or overbought in downtrend
            rsi_oversold = rsi_values[i] < 30
            rsi_overbought = rsi_values[i] > 70
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            
            if rsi_oversold and uptrend and vol_condition:
                signals[i] = 0.25
                position = 1
            elif rsi_overbought and not uptrend and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral or trend changes
            if rsi_values[i] > 50 or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral or trend changes
            if rsi_values[i] < 50 or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h RSI mean reversion with 1d EMA trend filter and volume confirmation
# - RSI < 30 in uptrend or > 70 in downtrend captures mean reversion within trend
# - 1d EMA(50) ensures alignment with higher timeframe trend
# - Volume spike (1.5x average) confirms participation
# - Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Position size 0.25 targets 20-40 trades/year, avoiding fee drag
# - Exit when RSI returns to neutral or trend changes provides logical exit