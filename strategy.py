#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
# RSI(14) < 30 for long, > 70 for short in ranging markets.
# 4h EMA(50) trend filter ensures we only trade counter-trend in strong trends.
# Volume spike (>1.5x 20-period average) confirms momentum exhaustion.
# Session filter (08-20 UTC) reduces noise.
# Designed for 15-30 trades/year to minimize fee drag in 1h timeframe.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
name = "1h_RSI_MeanReversion_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 1.5x average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND price above 4h EMA50 (uptrend) AND volume spike
            long_condition = rsi[i] < 30 and close[i] > ema_50_4h_aligned[i] and vol_spike
            # Short: RSI > 70 (overbought) AND price below 4h EMA50 (downtrend) AND volume spike
            short_condition = rsi[i] > 70 and close[i] < ema_50_4h_aligned[i] and vol_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) OR price below 4h EMA50 (trend change)
            exit_condition = rsi[i] > 50 or close[i] < ema_50_4h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) OR price above 4h EMA50 (trend change)
            exit_condition = rsi[i] < 50 or close[i] > ema_50_4h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals