#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI(14) with 1w EMA50 trend filter and 1d volume confirmation.
# Long when RSI crosses below 30 (oversold) AND 1d volume > 1.5x 20-period average AND close > 1w EMA50.
# Short when RSI crosses above 70 (overbought) AND 1d volume > 1.5x 20-period average AND close < 1w EMA50.
# Exit when RSI crosses back above 50 (for long) or below 50 (for short) to capture mean reversion.
# Uses RSI for mean reversion on daily timeframe with weekly trend filter to avoid counter-trend trades.
# Target: 40-80 total trades over 4 years (10-20/year) for low fee drift.

name = "1d_RSI14_1wEMA50_Volume"
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
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 1d data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI crosses below 30, volume spike, above 1w EMA50
            long_cond = (rsi[i] < 30) and (rsi[i-1] >= 30) and volume_filter[i] and (close[i] > ema50_1w_aligned[i])
            # Short conditions: RSI crosses above 70, volume spike, below 1w EMA50
            short_cond = (rsi[i] > 70) and (rsi[i-1] <= 70) and volume_filter[i] and (close[i] < ema50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 50 (mean reversion signal)
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses back below 50 (mean reversion signal)
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals