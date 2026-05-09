#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyKAMA_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    # ER = Efficiency Ratio
    change = abs(close_1w.diff(10))
    volatility = close_1w.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    # SC = Smoothing Constant
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros(len(close_1w))
    kama[0] = close_1w.iloc[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1w.iloc[i] - kama[i-1])
    kama = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Daily RSI for entry signal
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Daily volume confirmation
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.2 * vol_ma20[i]
        
        if position == 0:
            # Long: Price above weekly KAMA, RSI > 50, volume confirmation
            if close[i] > kama_1w_aligned[i] and rsi[i] > 50 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly KAMA, RSI < 50, volume confirmation
            elif close[i] < kama_1w_aligned[i] and rsi[i] < 50 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly KAMA or RSI < 40
            if close[i] < kama_1w_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly KAMA or RSI > 60
            if close[i] > kama_1w_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals