# 1d_1W_KAMA_Trend_Volume_Hypothesis
# Hypothesis: On daily chart, KAMA adapts to market efficiency, providing reliable trend direction.
# Price above/below KAMA with volume confirmation indicates institutional trend.
# Weekly trend filter (price vs weekly KAMA) ensures alignment with higher timeframe momentum,
# reducing whipsaws in ranging markets. Works in bull (follows uptrend) and bear (follows downtrend)
# by adapting to changing market conditions. Volume filters reduce false signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 1.0
        else:
            sum_change = np.sum(change[max(0, i-9):i+1])
            sum_vol = np.sum(volatility[max(0, i-9):i+1])
            er[i] = sum_change / sum_vol if sum_vol != 0 else 1.0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    change_w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility_w = np.abs(np.diff(close_1w))
    er_w = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i == 0:
            er_w[i] = 1.0
        else:
            sum_change_w = np.sum(change_w[max(0, i-9):i+1])
            sum_vol_w = np.sum(volatility_w[max(0, i-9):i+1])
            er_w[i] = sum_change_w / sum_vol_w if sum_vol_w != 0 else 1.0
    sc_w = (er_w * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_w[i] * (close_1w[i] - kama_1w[i-1])
    
    # Align indicators to lower timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above daily KAMA AND above weekly KAMA with volume spike
            if close[i] > kama_1d_aligned[i] and close[i] > kama_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below daily KAMA AND below weekly KAMA with volume spike
            elif close[i] < kama_1d_aligned[i] and close[i] < kama_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below daily KAMA or volume dies
            if close[i] < kama_1d_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above daily KAMA or volume dies
            if close[i] > kama_1d_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_KAMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0