#!/usr/bin/env python3
# 4h_rsi_ema_crossover_1d_trend_volume_v3
# Hypothesis: RSI(14) EMA(21) crossover on 4h filtered by 1d EMA50 trend and volume confirmation (1.5x avg).
# Long when RSI crosses above EMA with uptrend (price > 1d EMA50) and volume > 1.5x average.
# Short when RSI crosses below EMA with downtrend (price < 1d EMA50) and volume > 1.5x average.
# Uses momentum confirmation with trend filter to reduce whipsaw. Target: 25-40 trades/year (~100-160 total).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_ema_crossover_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0).values
    
    # Calculate EMA(21) of RSI
    rsi_ema = pd.Series(rsi).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(rsi_ema[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below EMA OR trend turns against us
            if (rsi[i] < rsi_ema[i]) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above EMA OR trend turns against us
            if (rsi[i] > rsi_ema[i]) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: RSI crosses above EMA with uptrend and volume confirmation
            if (rsi[i] > rsi_ema[i]) and (rsi[i-1] <= rsi_ema[i-1]) and (close[i] > ema_50_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI crosses below EMA with downtrend and volume confirmation
            elif (rsi[i] < rsi_ema[i]) and (rsi[i-1] >= rsi_ema[i-1]) and (close[i] < ema_50_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals