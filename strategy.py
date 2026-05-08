#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d for direction and 1h for entry timing.
# Uses 4h EMA for trend, 1d RSI for momentum, and volume confirmation.
# Long when 4h EMA up, 1d RSI > 50, 1h close > open, and volume > 1.5x average.
# Short when 4h EMA down, 1d RSI < 50, 1h close < open, and volume > 1.5x average.
# Flat otherwise. Uses 4h/1d for signal direction (low trade frequency), 1h only for entry timing.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Session filter: 08-20 UTC to reduce noise.
# Position size: 0.20 (discrete to minimize churn).

name = "1h_4hEMA_1dRSI_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract arrays
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 4h EMA(20) for trend
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_20_4h_up = ema_20_4h_aligned > np.roll(ema_20_4h_aligned, 1)
    ema_20_4h_up = np.where(np.isnan(ema_20_4h_up), False, ema_20_4h_up)
    
    # 1d RSI(14) for momentum
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[:14]) if len(gain) >= 14 else 0
    avg_loss[14] = np.mean(loss[:14]) if len(loss) >= 14 else 0
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([[np.nan], rsi_1d])
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if not in session or critical data missing
        if not in_session[i] or \
           np.isnan(ema_20_4h_up[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: 4h EMA up, 1d RSI > 50, bullish candle, volume spike
            if (ema_20_4h_up[i] and 
                rsi_1d_aligned[i] > 50 and 
                close[i] > open_price[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: 4h EMA down, 1d RSI < 50, bearish candle, volume spike
            elif (not ema_20_4h_up[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  close[i] < open_price[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend reversal or momentum loss
            if not ema_20_4h_up[i] or rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend reversal or momentum loss
            if ema_20_4h_up[i] or rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals