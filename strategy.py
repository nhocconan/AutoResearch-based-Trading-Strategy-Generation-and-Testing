#!/usr/bin/env python3
# 4h_rsi_ema_crossover_1d_trend_volume_v4
# Hypothesis: RSI mean reversion on 4h filtered by 1d EMA trend and volume confirmation.
# Long when RSI < 30 (oversold) with uptrend (price > 1d EMA50) and volume > 1.5x average.
# Short when RSI > 70 (overbought) with downtrend (price < 1d EMA50) and volume > 1.5x average.
# Designed to capture reversals in trending markets with strong volume. Target: 25-35 trades/year (~100-140 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_ema_crossover_1d_trend_volume_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) OR trend turns against us
            if (rsi[i] > 50) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) OR trend turns against us
            if (rsi[i] < 50) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: RSI < 30 (oversold) with uptrend and volume confirmation
            if (rsi[i] < 30) and (close[i] > ema_50_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 (overbought) with downtrend and volume confirmation
            elif (rsi[i] > 70) and (close[i] < ema_50_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals