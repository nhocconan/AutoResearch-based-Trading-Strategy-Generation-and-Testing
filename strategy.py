#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and volume spike.
Works in bull/bear markets because: 1) RSI <30/70 identifies overbought/oversold conditions, 
2) 4h EMA50 ensures trend alignment to avoid counter-trend trades, 
3) Volume spike confirms reversal strength. Target 15-35 trades/year.
"""
name = "1h_RSI_MeanReversion_4hTrend_VolumeSpike"
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
    
    # === 4h DATA FOR TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # 50 for 4h EMA, 14 for RSI, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI < 30 (oversold), price above 4h EMA50, volume spike
            if (rsi[i] < 30 and 
                close[i] > ema50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: RSI > 70 (overbought), price below 4h EMA50, volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: RSI > 50 or price below 4h EMA50
            if (rsi[i] > 50) or (close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI < 50 or price above 4h EMA50
            if (rsi[i] < 50) or (close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals