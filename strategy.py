#!/usr/bin/env python3
# 1d_RSI_Trend_Pullback_1wTrend_Volume
# Hypothesis: Uses RSI pullbacks on daily timeframe aligned with 1-week trend and volume confirmation.
# Long when RSI < 30 (oversold) and price above 1w EMA20, short when RSI > 70 (overbought) and price below 1w EMA20.
# Volume filter requires volume > 1.5x 20-day average to confirm momentum.
# Designed for 1d to work in both bull and bear markets via trend filter and mean-reversion entries.
# Targets 20-40 trades per year to minimize fee drag.

name = "1d_RSI_Trend_Pullback_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate volume spike: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price above 1w EMA20 + volume spike
            if rsi[i] < 30 and close[i] > ema_20_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + price below 1w EMA20 + volume spike
            elif rsi[i] > 70 and close[i] < ema_20_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses above 50 (momentum shift) or price closes below 1w EMA20
            if rsi[i] > 50 or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses below 50 (momentum shift) or price closes above 1w EMA20
            if rsi[i] < 50 or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals