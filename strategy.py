#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily RSI(14) mean-reversion with weekly trend filter and volume confirmation.
# RSI < 30 = oversold (long), RSI > 70 = overbought (short) only when aligned with weekly trend.
# Weekly EMA20 trend filter ensures trades follow higher timeframe momentum.
# Volume > 1.5x 20-day average confirms institutional participation.
# Target: 15-25 trades/year to minimize fee drag while capturing mean-reversion edges.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily closes
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # Wilder's smoothing
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan  # Not enough data for first 14 periods
    
    # Align RSI to daily timeframe (already aligned, just need to map)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: RSI oversold (<30) with weekly uptrend and volume confirmation
        if (rsi_1d_aligned[i] < 30 and 
            close[i] > ema20_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short: RSI overbought (>70) with weekly downtrend and volume confirmation
        elif (rsi_1d_aligned[i] > 70 and 
              close[i] < ema20_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit: RSI returns to neutral zone (40-60) or opposite extreme
        elif position == 1 and (rsi_1d_aligned[i] > 50 or rsi_1d_aligned[i] > 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi_1d_aligned[i] < 50 or rsi_1d_aligned[i] < 30):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_RSI14_MeanReversion_1wEMA20_VolumeFilter"
timeframe = "1d"
leverage = 1.0