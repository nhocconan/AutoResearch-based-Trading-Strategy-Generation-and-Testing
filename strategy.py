#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly RSI filter and daily volume confirmation
# - Long when price breaks above 20-day high + weekly RSI < 60 (not overbought) + volume > 1.5x 20-day average
# - Short when price breaks below 20-day low + weekly RSI > 40 (not oversold) + volume > 1.5x 20-day average
# - Uses weekly RSI to avoid buying into strength or selling into weakness
# - Designed for 1d timeframe to capture major moves with low trade frequency
# - Target: 15-25 trades/year to minimize fee drag

name = "1d_Donchian20_WeeklyRSI_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.concatenate([np.full(13, np.nan), rsi_1w[13:]])  # align with weekly bars
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: price breaks above 20-day high + weekly RSI not overbought + volume
            if close[i] > highest_high[i] and rsi_1w_aligned[i] < 60 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below 20-day low + weekly RSI not oversold + volume
            elif close[i] < lowest_low[i] and rsi_1w_aligned[i] > 40 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on break below 20-day low or weekly RSI overbought
            if close[i] < lowest_low[i] or rsi_1w_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on break above 20-day high or weekly RSI oversold
            if close[i] > highest_high[i] or rsi_1w_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals