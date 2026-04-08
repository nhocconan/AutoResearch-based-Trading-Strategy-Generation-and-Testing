#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_ema_trend_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend (primary trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for long-term trend
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get 1d data for Williams fractals (key support/resistance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams fractals: need 5 bars (2 left, center, 2 right)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n] is highest of [n-2, n-1, n, n+1, n+2]
    bearish = np.zeros(len(high_1d), dtype=bool)
    for i in range(2, len(high_1d)-2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish[i] = True
    
    # Bullish fractal: low[n] is lowest of [n-2, n-1, n, n+1, n+2]
    bullish = np.zeros(len(low_1d), dtype=bool)
    for i in range(2, len(low_1d)-2):
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish[i] = True
    
    # Convert to price levels (use the fractal high/low as resistance/support)
    bearish_level = np.where(bearish, high_1d, np.nan)
    bullish_level = np.where(bullish, low_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    bearish_series = pd.Series(bearish_level)
    bullish_series = pd.Series(bullish_level)
    bearish_ffilled = bearish_series.ffill().values
    bullish_ffilled = bullish_series.ffill().values
    
    # Align to 6h with 2-bar delay for fractal confirmation (needs 2 future 1d bars)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_ffilled, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_ffilled, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.5x average of last 12 periods (12*6h = 3 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to bullish fractal support or trend changes
            if close[i] <= bullish_aligned[i] or ema_1w_aligned[i] < ema_1w_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to bearish fractal resistance or trend changes
            if close[i] >= bearish_aligned[i] or ema_1w_aligned[i] > ema_1w_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above bearish fractal resistance with volume and 1w uptrend
            if (not np.isnan(bearish_aligned[i]) and close[i] > bearish_aligned[i] and 
                ema_1w_aligned[i] > ema_1w_aligned[max(0, i-1)] and  # Uptrend confirmation
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below bullish fractal support with volume and 1w downtrend
            elif (not np.isnan(bullish_aligned[i]) and close[i] < bullish_aligned[i] and 
                  ema_1w_aligned[i] < ema_1w_aligned[max(0, i-1)] and  # Downtrend confirmation
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals