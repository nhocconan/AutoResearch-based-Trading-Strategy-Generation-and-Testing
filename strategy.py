#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Trend_Momentum
Hypothesis: Uses KAMA trend direction on 1d with RSI momentum on 1w for high-probability entries.
Trades only when KAMA confirms trend and RSI shows momentum, with volatility filter to avoid chop.
Designed for low trade frequency (10-25/year) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Trend_Momentum"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on daily close (trend indicator)
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = [close[0]]
    for i in range(1, n):
        kama.append(kama[-1] + sc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # Load 1w data ONCE before loop for RSI momentum
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate RSI on weekly close
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where(np.isnan(rsi_1w), 50, rsi_1w)
    
    # Align RSI to daily timeframe (wait for weekly close)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volatility filter: avoid choppy markets
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_ratio = atr / (pd.Series(atr).rolling(window=50, min_periods=50).mean().values + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price relative to KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # Momentum filter: RSI showing strength
        rsi_bullish = rsi_1w_aligned[i] > 55
        rsi_bearish = rsi_1w_aligned[i] < 45
        
        # Volatility filter: avoid high churn periods
        vol_filter = atr_ratio[i] < 1.5
        
        # Entry conditions
        long_entry = above_kama and rsi_bullish and vol_filter
        short_entry = below_kama and rsi_bearish and vol_filter
        
        # Exit conditions: opposite signal or volatility expansion
        long_exit = (below_kama or not rsi_bullish or atr_ratio[i] > 2.0)
        short_exit = (above_kama or not rsi_bearish or atr_ratio[i] > 2.0)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals