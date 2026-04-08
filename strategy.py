#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_trend_volume_momentum"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for momentum filter (RSI14)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate RSI(14) on weekly close
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Volume threshold: 4h volume > 1.5x daily average volume (scaled)
    # 1 day = 6 x 4h bars, so divide daily avg by 6
    vol_threshold = vol_avg_1d_aligned / 6.0
    vol_confirm = volume > vol_threshold * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any indicator not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_14_1w_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1d EMA50 OR weekly RSI < 40 (losing momentum)
            if close[i] < ema_50_1d_aligned[i] or rsi_14_1w_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d EMA50 OR weekly RSI > 60 (losing momentum)
            if close[i] > ema_50_1d_aligned[i] or rsi_14_1w_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above 1d EMA50 AND weekly RSI > 50 (bullish momentum) + volume confirmation
            if close[i] > ema_50_1d_aligned[i] and rsi_14_1w_aligned[i] > 50 and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below 1d EMA50 AND weekly RSI < 50 (bearish momentum) + volume confirmation
            elif close[i] < ema_50_1d_aligned[i] and rsi_14_1w_aligned[i] < 50 and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals