#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h volume-weighted RSI with 4h trend filter and daily volume regime.
# Uses RSI(14) on volume-weighted close (VWC) to reduce noise and identify momentum.
# Long when VWC RSI < 30 (oversold) + 4h EMA(50) up + daily volume > 20-day average.
# Short when VWC RSI > 70 (overbought) + 4h EMA(50) down + daily volume > 20-day average.
# Designed to capture mean reversion in ranging markets while filtering with higher timeframe trend.
# Volume-weighted RSI is less prone to whipsaw than standard RISI.
name = "1h_VolumeWeightedRSI_4hEMA50_DailyVol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume-weighted close: (high + low + close + volume) / 4
    vwc = (high + low + close + volume) / 4
    
    # RSI(14) on VWC
    delta = np.diff(vwc, prepend=vwc[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Daily volume regime: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_20d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    vol_regime = volume > (1.5 * vol_20d_aligned)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_20d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VWC RSI < 30 (oversold) + 4h EMA up + volume regime
            if (rsi[i] < 30 and close[i] > ema_4h_aligned[i] and vol_regime[i]):
                signals[i] = 0.20
                position = 1
            # Short: VWC RSI > 70 (overbought) + 4h EMA down + volume regime
            elif (rsi[i] > 70 and close[i] < ema_4h_aligned[i] and vol_regime[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: VWC RSI > 50 or price below 4h EMA
            if rsi[i] > 50 or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: VWC RSI < 50 or price above 4h EMA
            if rsi[i] < 50 or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals