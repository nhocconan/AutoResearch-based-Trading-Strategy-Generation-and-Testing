#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and volume spike confirmation.
- Long when RSI(14) < 30 AND price > 4h EMA50 (bullish trend) AND volume > 2.0 * median volume (oversold in uptrend)
- Short when RSI(14) > 70 AND price < 4h EMA50 (bearish trend) AND volume > 2.0 * median volume (overbought in downtrend)
- Exit on opposite RSI extreme or trend reversal (price crosses 4h EMA50)
- Uses 1h primary timeframe with 4h HTF for trend direction to reduce whipsaws
- Volume spike filters low-momentum breakouts, focusing on high-conviction mean reversion
- Designed for BTC/ETH with edge in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
- Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
"""

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
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 2.0 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold, trend up, volume confirmation
            if rsi[i] < 30 and close[i] > ema_50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought, trend down, volume confirmation
            elif rsi[i] > 70 and close[i] < ema_50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI overbought OR trend reversal (price < EMA50)
            if rsi[i] > 70 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI oversold OR trend reversal (price > EMA50)
            if rsi[i] < 30 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0