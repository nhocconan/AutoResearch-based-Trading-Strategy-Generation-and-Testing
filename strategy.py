#!/usr/bin/env python3
name = "12h_1w_1d_KAMA_RSI_Trend_Volume"
timeframe = "12h"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily KAMA (ER=10)
    close_1d = pd.Series(df_1d['close'])
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff(1).abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = [close_1d.iloc[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily RSI(14)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Volume spike detection: 4-period average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14)  # Wait for weekly EMA and daily RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, weekly uptrend, volume spike
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and weekly_uptrend and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, weekly downtrend, volume spike
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and not weekly_uptrend and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below KAMA or RSI < 40 or volume drops
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40 or volume[i] < vol_ma_4[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above KAMA or RSI > 60 or volume drops
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60 or volume[i] < vol_ma_4[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h KAMA/RSI with weekly trend filter and volume confirmation
# - KAMA adapts to market noise, reducing false signals in sideways markets
# - RSI(14) filters for momentum (long when RSI>50, short when RSI<50)
# - Weekly EMA(34) ensures alignment with higher timeframe trend
# - Volume spike (1.5x average) confirms institutional participation
# - Dynamic exits based on KAMA cross and RSI extremes prevent whipsaws
# - Works in bull markets (KAMA up, RSI>50, weekly uptrend) and bear markets (reverse)
# - Position size 0.25 targets ~25-40 trades/year, avoiding fee drag
# - Novel combination: KAMA (1d) + RSI (1d) + weekly trend (1w) + volume (12h) not recently tried
# - Designed to reduce trade frequency while maintaining edge in both regimes