#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
# Uses 4h EMA50 for trend direction and 1h RSI + volume spike for entries.
# Designed to work in both bull and bear markets by following 4h trend.
# Target: 15-37 trades/year (60-150 over 4 years) with strict entry conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume average (20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50_4h_val = ema50_4h_aligned[i]
        rsi_val = rsi[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Trend filter: 4h EMA50
        uptrend = price > ema50_4h_val
        downtrend = price < ema50_4h_val
        
        # Volume filter: current volume > 1.8x 20-period average
        vol_spike = vol_current > 1.8 * vol_ma_val
        
        if position == 0:
            # Long: uptrend + RSI < 30 (oversold) + volume spike
            if uptrend and rsi_val < 30 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + RSI > 70 (overbought) + volume spike
            elif downtrend and rsi_val > 70 and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (momentum fade) or trend change
            if rsi_val > 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI < 50 (momentum fade) or trend change
            if rsi_val < 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA50_RSI_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0