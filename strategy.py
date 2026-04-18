#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily volume confirmation and 1h EMA trend filter.
# Uses Camarilla levels (H3/L3) from prior day for institutional breakout levels.
# Requires volume > 1.5x 20-period average for conviction.
# Uses 1h EMA(34) as trend filter to avoid counter-trend trades.
# Designed for low trade frequency (20-40/year) with clear entry/exit rules.
# Works in bull markets (long breaks above H3) and bear markets (short breaks below L3).
name = "4h_Camarilla_H3L3_Volume_EMA34_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    camarilla_width = 1.1 * (high_prev - low_prev) / 2
    H3 = close_prev + camarilla_width
    L3 = close_prev - camarilla_width
    
    # Align Camarilla levels to 4h timeframe (available after daily bar closes)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Get 1h data for EMA trend filter
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Calculate EMA(34) on 1h closes
    ema_34_1h = pd.Series(close_1h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1h EMA to 4h timeframe
    ema_34_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_34_1h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_34_1h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below 1h EMA34
        price_vs_ema = close[i] - ema_34_1h_aligned[i]
        
        if position == 0:
            # Long: price breaks above H3 AND volume confirmation AND uptrend (price > EMA)
            long_breakout = close[i] > H3_aligned[i]
            if vol_confirm and (price_vs_ema > 0) and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume confirmation AND downtrend (price < EMA)
            elif vol_confirm and (price_vs_ema < 0) and close[i] < L3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR trend turns against (price < EMA)
            exit_condition = close[i] < L3_aligned[i] or (price_vs_ema < 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR trend turns against (price > EMA)
            exit_condition = close[i] > H3_aligned[i] or (price_vs_ema > 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals