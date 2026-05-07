#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Wilder's RSI with 14-period, combined with 1w EMA50 trend filter and volume confirmation.
# Long when RSI < 30 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-period average.
# Short when RSI > 70 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-period average.
# Exit when RSI crosses back to neutral (40 for long exit, 60 for short exit).
# Designed for 1d timeframe with low trade frequency (target: 10-25/year) to avoid fee drag.
# Uses 1w EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
name = "1d_RSI14_1wEMA50_VolumeFilter"
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
    
    # Wilder's RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # For periods where avg_loss is 0, RSI = 100
    rsi[avg_loss == 0] = 100
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI < 30 (oversold), price > 1w EMA50 (uptrend), volume filter
            long_cond = (rsi[i] < 30) and (close[i] > ema50_1w_aligned[i]) and volume_filter[i]
            # Short conditions: RSI > 70 (overbought), price < 1w EMA50 (downtrend), volume filter
            short_cond = (rsi[i] > 70) and (close[i] < ema50_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 40 (mean reversion)
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses back below 60 (mean reversion)
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals