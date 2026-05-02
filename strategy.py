#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 4h timeframe for optimal trade frequency (target: 75-200 trades over 4 years)
# Donchian breakouts capture momentum; 1d EMA34 ensures higher timeframe trend alignment
# Volume confirmation filters low-quality breakouts (volume > 1.5x 20-period average)
# Works in both bull and bear markets via trend-following logic
# Discrete position sizing: 0.25 (25% of capital) minimizes fee churn while maintaining exposure
# ATR-based stoploss would be handled externally per engine rules (exit via signal=0)

name = "4h_Donchian20_1dEMA34_VolumeConfirm_Trend"
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
    
    # Calculate 1d Donchian channels (20-period) - using 1d as HTF for structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels: upper = rolling max(high, 20), lower = rolling min(low, 20)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume average for confirmation (on 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(34, 20, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 4h Donchian high AND price > 1d EMA34 (bullish trend) 
            # AND volume > 1.5x 20-period volume average (confirmation)
            if (close[i] > donch_high_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 4h Donchian low AND price < 1d EMA34 (bearish trend)
            # AND volume > 1.5x 20-period volume average (confirmation)
            elif (close[i] < donch_low_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below 4h Donchian low OR below 1d EMA34 (trend change)
            if close[i] < donch_low_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above 4h Donchian high OR above 1d EMA34 (trend change)
            if close[i] > donch_high_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals