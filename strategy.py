#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + RSI filter
# Uses Donchian channel breakouts for trend capture, volume to confirm breakout strength,
# and RSI(14) to avoid overbought/oversold extremes. Works in both bull and bear by
# only taking breakouts in the direction of the 4h trend (EMA50).
# Target: 80-180 total trades over 4 years (20-45/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on 4h for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI(14) on 1d
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + RSI not overbought + price above EMA50
        if (close[i] > donch_high_4h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            rsi_aligned[i] < 70 and
            close[i] > ema50_4h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + RSI not oversold + price below EMA50
        elif (close[i] < donch_low_4h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              rsi_aligned[i] > 30 and
              close[i] < ema50_4h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or RSI extreme
        elif position == 1 and (close[i] < donch_low_4h_aligned[i] or rsi_aligned[i] > 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donch_high_4h_aligned[i] or rsi_aligned[i] < 30):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Volume_RSI_Filter"
timeframe = "4h"
leverage = 1.0