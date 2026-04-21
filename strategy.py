#!/usr/bin/env python3
"""
4h_RSI_MeanReversion_BollingerBand_Extremes
Hypothesis: In both bull and bear markets, RSI extremes (>80 or <20) combined with price outside Bollinger Bands (2,2) signal exhaustion and mean reversion. Trades are taken in the direction of the 1d trend (using EMA50) to avoid counter-trend whipsaw. Works in ranging markets as well as during pullbacks in trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        upper = upper_band[i]
        lower = lower_band[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: RSI < 20 (oversold) + price < lower BB + price > 1d EMA50 (uptrend filter)
            if rsi_val < 20 and price < lower and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 80 (overbought) + price > upper BB + price < 1d EMA50 (downtrend filter)
            elif rsi_val > 80 and price > upper and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion) or price > SMA20
            if rsi_val > 50 or price > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion) or price < SMA20
            if rsi_val < 50 or price < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_MeanReversion_BollingerBand_Extremes"
timeframe = "4h"
leverage = 1.0