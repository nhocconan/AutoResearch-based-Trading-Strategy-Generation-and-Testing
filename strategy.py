#!/usr/bin/env python3
"""
4h_rsi_ema_crossover_1d_trend_volume_v1
Hypothesis: Use RSI + EMA cross on 4h for momentum, confirmed by 1d trend and volume.
- Entry: RSI crosses above 50 + EMA(9) > EMA(21) + 1d close above EMA(50) + volume > 1.5x avg
- Exit: RSI crosses below 50 or EMA(9) < EMA(21)
- Volume filter to avoid false breakouts
Target: 20-50 trades/year (80-200 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_ema_crossover_1d_trend_volume_v1"
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
    
    # 4h indicators
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # EMA(9) and EMA(21)
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    trend_down_1d = close_1d < ema50_1d
    
    # Forward fill 1d trend
    trend_up_1d_series = pd.Series(trend_up_1d)
    trend_down_1d_series = pd.Series(trend_down_1d)
    trend_up_1d_ffilled = trend_up_1d_series.ffill().values
    trend_down_1d_ffilled = trend_down_1d_series.ffill().values
    
    # Align 1d trend to 4h
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d_ffilled)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d_ffilled)
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI < 50 or EMA9 < EMA21
            if rsi[i] < 50 or ema9[i] < ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI > 50 or EMA9 > EMA21
            if rsi[i] > 50 or ema9[i] > ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI crosses above 50 + EMA9 > EMA21 + 1d uptrend + volume
            if (rsi[i] > 50 and rsi[i-1] <= 50 and 
                ema9[i] > ema21[i] and 
                trend_up_1d_aligned[i] and 
                volume_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI crosses below 50 + EMA9 < EMA21 + 1d downtrend + volume
            elif (rsi[i] < 50 and rsi[i-1] >= 50 and 
                  ema9[i] < ema21[i] and 
                  trend_down_1d_aligned[i] and 
                  volume_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals