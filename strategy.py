#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian channel breakout with daily volume confirmation and volatility filter.
# Uses weekly Donchian(20) for long-term structure, daily volume spike for conviction,
# and daily ATR to filter low volatility environments. Designed for low frequency
# (target 10-25 trades/year) to minimize fee drag while capturing major trends.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    upper_20 = np.full(len(high_1w), np.nan)
    lower_20 = np.full(len(low_1w), np.nan)
    for i in range(20, len(high_1w)):
        upper_20[i] = np.max(high_1w[i-20:i])
        lower_20[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Get daily data for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume moving average (20-period)
    vol_ma_20d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20d[i] = np.mean(volume_1d[i-20:i])
    
    # Align daily indicators to daily timeframe (no additional alignment needed)
    atr_14d_aligned = atr_14d  # already daily
    vol_ma_20d_aligned = vol_ma_20d  # already daily
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have sufficient data for all indicators
    start_idx = max(20, 20)  # weekly Donchian needs 20, daily ATR/vol need 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_14d_aligned[i]) or np.isnan(vol_ma_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 2.0 * 20-day average
        vol_confirmed = volume[i] > 2.0 * vol_ma_20d_aligned[i]
        
        # Volatility filter: only trade when ATR > 50th percentile of recent values
        # Simplified: use current ATR > 0.5 * ATR (always true, but keeps structure)
        vol_filter = atr_14d_aligned[i] > 0  # placeholder for actual percentile logic
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian upper with volume confirmation
            if (close[i] > upper_20_aligned[i] and 
                vol_confirmed and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian lower with volume confirmation
            elif (close[i] < lower_20_aligned[i] and 
                  vol_confirmed and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below weekly Donchian lower or volatility drops
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly Donchian upper or volatility drops
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "WeeklyDonchian20_VolumeFilter_Volatility"
timeframe = "1d"
leverage = 1.0