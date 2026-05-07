#!/usr/bin/env python3
name = "1d_KAMA_RSI_Trend_v2"
timeframe = "1d"
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
    
    # Get weekly data for trend filter (primary filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly KAMA for trend direction (more adaptive than EMA)
    close_1w = df_1w['close']
    # Calculate efficiency ratio
    change = abs(close_1w.diff(10))
    volatility = close_1w.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(len(close_1w))
    kama[0] = close_1w.iloc[0]
    for i in range(1, len(close_1w)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close_1w.iloc[i] - kama[i-1])
    kama_1w = kama
    
    # Align weekly KAMA to daily
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI for entry timing (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 for volume MA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly KAMA AND RSI < 40 (oversold) + volume
            if close[i] > kama_1w_aligned[i] and rsi_values[i] < 40 and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly KAMA AND RSI > 60 (overbought) + volume
            elif close[i] < kama_1w_aligned[i] and rsi_values[i] > 60 and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses weekly KAMA or RSI reaches extreme
            if position == 1:
                if close[i] < kama_1w_aligned[i] or rsi_values[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_1w_aligned[i] or rsi_values[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals