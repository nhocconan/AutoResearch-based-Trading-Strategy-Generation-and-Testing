#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Trend_RSI_MeanRev_2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter (ER=10, fast=2, slow=30)
    close_1w = pd.Series(df_1w['close'].values)
    # Efficiency Ratio
    change = abs(close_1w.diff(10))
    volatility = close_1w.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w.iloc[0]
    for i in range(1, len(close_1w)):
        if not np.isnan(sc.iloc[i]):
            kama_1w[i] = kama_1w[i-1] + sc.iloc[i] * (close_1w.iloc[i] - kama_1w[i-1])
        else:
            kama_1w[i] = kama_1w[i-1]
    kama_1w = kama_1w[~np.isnan(kama_1w)] if np.any(np.isnan(kama_1w)) else kama_1w
    # Ensure same length
    if len(kama_1w) != len(close_1w):
        kama_1w = pd.Series(kama_1w).reindex_like(close_1w, method='ffill').bfill().values
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI for mean reversion (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Daily volume spike detection
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: KAMA uptrend + RSI oversold + volume
            if close[i] > kama_1w_aligned[i] and rsi[i] < 30 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend + RSI overbought + volume
            elif close[i] < kama_1w_aligned[i] and rsi[i] > 70 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend reversal
            if rsi[i] > 70 or close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or trend reversal
            if rsi[i] < 30 or close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals