#!/usr/bin/env python3
"""
1d_1w_Keltner_CCI_Breakout_v1
Hypothesis: Daily breakout of Keltner Channel (2x ATR) with CCI(20) momentum confirmation and weekly trend filter (price > weekly SMA50). Designed to capture sustained trends in both bull and bear markets with low frequency (target: 30-100 trades over 4 years). Uses discrete position sizing (0.25) to minimize churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Keltner_CCI_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY TREND FILTER (SMA50) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    sma_50_1w = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # === DAILY KELTNER CHANNEL (2x ATR) ===
    # Calculate ATR(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA(20) of close for Keltner middle
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Upper/Lower bands
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    
    # === CCI(20) MOMENTUM ===
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(cci[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: Close above Keltner Upper + CCI > 100 + price above weekly SMA50
        long_signal = (close[i] > keltner_upper[i]) and (cci[i] > 100) and (close[i] > sma_50_1w_aligned[i])
        
        # Short: Close below Keltner Lower + CCI < -100 + price below weekly SMA50
        short_signal = (close[i] < keltner_lower[i]) and (cci[i] < -100) and (close[i] < sma_50_1w_aligned[i])
        
        # Exit: CCI crosses back through zero (mean reversion)
        exit_long = (position == 1) and (cci[i] < 0)
        exit_short = (position == -1) and (cci[i] > 0)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals