#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with weekly EMA trend filter and volume confirmation.
# Uses daily Donchian(20) for breakout signals, weekly EMA(34) for trend direction, and daily volume spike for confirmation.
# Designed for low trade frequency (target 20-50/year) to minimize fee drag in both bull and bear markets.
# The weekly EMA filter ensures we only trade in the direction of the higher timeframe trend,
# reducing whipsaw during sideways markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channel (20-period) on daily data
    donch_high = np.full(len(high_1d), np.nan)
    donch_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-20:i])
        donch_low[i] = np.min(low_1d[i-20:i])
    
    # Align daily Donchian levels to 1d timeframe (no shift needed as we use previous day's values)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate weekly EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily volume moving average (20-period)
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian channel
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 2.0 * 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        # Trend filter: price above weekly EMA34 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above daily Donchian high, with volume and trend filter
            if (close[i] > donch_high_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below daily Donchian low, with volume and trend filter
            elif (close[i] < donch_low_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below daily Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above daily Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0