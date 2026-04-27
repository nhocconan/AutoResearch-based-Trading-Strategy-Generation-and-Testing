# 100970 - 1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: Daily KAMA trend direction combined with RSI mean reversion and Choppiness index regime filter.
# Works in both bull and bear markets: KAMA adapts to trend, RSI captures reversals, Chop filter avoids whipsaws in ranging markets.
# Target: 15-25 trades/year to minimize fee drag. Uses discrete position sizing (0.25) to reduce churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Choppiness index (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Choppiness Index (14-period)
    atr_1w = []
    tr_1w = []
    for i in range(1, len(close_1w)):
        tr = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
        tr_1w.append(tr)
    tr_1w = np.array(tr_1w)
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    max_hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    chop_raw = 100 * np.log10(atr_1w.sum() / (max_hh - min_ll)) / np.log10(14)
    chop = pd.Series(chop_raw).fillna(50).values  # fill NaN with neutral 50
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period volatility
    er = np.zeros_like(close_1d)
    er[10:] = change[9:] / volatility[9:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # neutral before enough data
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chop > 50 = ranging (mean revert), Chop < 50 = trending (follow trend)
        is_ranging = chop_aligned[i] > 50
        
        if is_ranging:
            # In ranging markets: mean reversion at RSI extremes
            if rsi_aligned[i] < 30 and close[i] > kama_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif rsi_aligned[i] > 70 and close[i] < kama_aligned[i]:
                signals[i] = -0.25
                position = -1
            # Exit when RSI returns to neutral
            elif position == 1 and rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # In trending markets: follow KAMA direction
            if close[i] > kama_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < kama_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0