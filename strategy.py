#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d EMA34 trend alignment, 4h Donchian20 breakout,
# volume confirmation, and RSI filter for momentum confirmation. Uses tight entry
# conditions to limit trades (~30-40/year) and avoid fee drag. Works in bull/bear
# by following higher timeframe trends and avoiding choppy markets via EMA filter.
name = "4h_1d_EMA34_Donchian20_Volume_RSI"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Donchian20 breakout (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Donchian channels: 20-period high/low
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    # RSI(14) for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20_4h_aligned[i]) or 
            np.isnan(low_20_4h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(rsi_values[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d EMA34 AND breaks 4h Donchian high with volume and RSI>50
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > high_20_4h_aligned[i] and 
                volume_filter[i] and 
                rsi_values[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34 AND breaks 4h Donchian low with volume and RSI<50
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < low_20_4h_aligned[i] and 
                  volume_filter[i] and 
                  rsi_values[i] < 50):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d EMA34 or 4h Donchian low
            if close[i] < ema_34_1d_aligned[i] or close[i] < low_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d EMA34 or 4h Donchian high
            if close[i] > ema_34_1d_aligned[i] or close[i] > high_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals