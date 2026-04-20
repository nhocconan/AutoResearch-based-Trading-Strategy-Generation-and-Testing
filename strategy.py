#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_RSI_volatility_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === KAMA Calculation (Daily) ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of absolute changes
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align length
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(loss_ma > 0, gain_ma / loss_ma, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Weekly ATR for Volatility Filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === Weekly KAMA for Trend Filter ===
    direction_1w = np.abs(np.diff(close_1w, n=10))
    volatility_1w = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)
    volatility_1w = np.concatenate([np.full(9, np.nan), volatility_1w])
    er_1w = np.where(volatility_1w > 0, direction_1w / volatility_1w, 0)
    sc_1w = (er_1w * (2/2 - 2/30) + 2/30) ** 2
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if np.isnan(sc_1w[i]):
            kama_1w[i] = kama_1w[i-1]
        else:
            kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(kama_1w_aligned[i]) or np.isnan(atr_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        weekly_kama = kama_1w_aligned[i]
        weekly_atr = atr_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility regimes
        if weekly_atr < np.nanpercentile(atr_1w_aligned[:i+1], 20) if i >= 20 else False:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily KAMA, RSI not overbought, weekly trend up
            if close_val > kama_val and rsi_val < 70 and close_val > weekly_kama:
                signals[i] = 0.25
                position = 1
            # Short: price below daily KAMA, RSI not oversold, weekly trend down
            elif close_val < kama_val and rsi_val > 30 and close_val < weekly_kama:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below daily KAMA or RSI overbought
            if close_val < kama_val or rsi_val > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above daily KAMA or RSI oversold
            if close_val > kama_val or rsi_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals