#!/usr/bin/env python3
# 12h_rsi_extreme_trend_volume_v1
# Hypothesis: On 12h timeframe, use extreme RSI levels combined with daily trend and volume confirmation.
# Long when RSI < 20 (deep oversold) with daily trend up (price > daily EMA50) and volume > 1.5x average.
# Short when RSI > 80 (deep overbought) with daily trend down (price < daily EMA50) and volume > 1.5x average.
# Exit when RSI returns to neutral range (40-60) or volume drops below average.
# This strategy targets rare but high-probability mean-reversion setups in both bull and bear markets.
# Extreme RSI readings indicate exhaustion, and trend filter ensures we trade with higher timeframe momentum.
# Volume confirmation avoids false signals in low-liquidity periods.
# Expected trade frequency: 15-25 per year (60-100 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_rsi_extreme_trend_volume_v1"
timeframe = "12h"
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
    
    # Calculate RSI on 12h data
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rs[14:] = avg_gain[13:] / (avg_loss[13:] + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    daily_close = df_daily['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_12h = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    # Volume confirmation: 20-period average on 12h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after RSI warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi[i]) or np.isnan(daily_ema50_12h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral range (40-60) or volume drops below average
            if rsi[i] >= 40 and rsi[i] <= 60 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral range (40-60) or volume drops below average
            if rsi[i] >= 40 and rsi[i] <= 60 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema50_12h[i]
            daily_downtrend = close[i] < daily_ema50_12h[i]
            
            # Long entry: RSI < 20 (deep oversold) with volume and uptrend
            if rsi[i] < 20 and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 80 (deep overbought) with volume and downtrend
            elif rsi[i] > 80 and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals