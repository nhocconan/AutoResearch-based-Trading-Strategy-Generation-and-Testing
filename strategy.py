# 1d_KAMA_1wTrend_RSI
# 1d timeframe, 1w trend filter, RSI momentum + volume confirmation
# Uses KAMA (adaptive moving average) to follow price trends with low lag in trends and high lag in ranges.
# 1w KAMA direction filters trades to follow higher timeframe trend.
# RSI(14) > 55 for long, < 45 for short with volume > 1.5x 20-bar average.
# Designed for low trade frequency (< 50/year) to minimize fee drag in ranging markets.
# Works in bull/bear by following 1w trend and avoiding counter-trend trades.

name = "1d_KAMA_1wTrend_RSI"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1w KAMA for trend direction ===
    close_1w = df_1w['close'].values
    # Efficiency ratio (ER)
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w, prepend=close_1w[0])), axis=0) if False else None  # placeholder
    # Proper ER calculation
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    # Rolling sum for volatility
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    change_val = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    # Avoid division by zero
    er = np.where(vol_sum > 0, change_val / vol_sum, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # === Daily indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        close_val = close[i]
        kama_val = kama_1w_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 1w KAMA (uptrend) + RSI > 55 + volume confirmation
            if close_val > kama_val and rsi_val > 55 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price below 1w KAMA (downtrend) + RSI < 45 + volume confirmation
            elif close_val < kama_val and rsi_val < 45 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below 1w KAMA or RSI < 40
            if close_val < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above 1w KAMA or RSI > 60
            if close_val > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals