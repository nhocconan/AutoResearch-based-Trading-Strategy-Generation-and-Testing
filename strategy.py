#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Candlestick patterns and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Daily range for candle body size
    body_size_1d = np.abs(close_1d - open_1d)
    range_1d = high_1d - low_1d
    
    # Daily 20-period SMA for Bollinger Bands
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align 1d data to 4h
    body_size_1d_aligned = align_htf_to_ltf(prices, df_1d, body_size_1d)
    range_1d_aligned = align_htf_to_ltf(prices, df_1d, range_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d.values)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d.values)
    
    # Align weekly EMA50 to 4h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w.values)
    
    # Daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(body_size_1d_aligned[i]) or np.isnan(range_1d_aligned[i]) or
            np.isnan(upper_bb_1d_aligned[i]) or np.isnan(lower_bb_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_condition = volume_1d_aligned[i] > (volume_ma_20_1d_aligned[i] * 1.5)
        
        # Trend filter: only long when price > weekly EMA50, short when price < weekly EMA50
        long_trend = close[i] > ema_50_1w_aligned[i]
        short_trend = close[i] < ema_50_1w_aligned[i]
        
        # Candlestick pattern: small body (doji-like) near Bollinger Bands
        # Long: small body near lower BB (potential reversal up)
        # Short: small body near upper BB (potential reversal down)
        body_ratio = body_size_1d_aligned[i] / range_1d_aligned[i] if range_1d_aligned[i] > 0 else 1
        near_lower_bb = close[i] < lower_bb_1d_aligned[i] + (0.1 * (upper_bb_1d_aligned[i] - lower_bb_1d_aligned[i]))
        near_upper_bb = close[i] > upper_bb_1d_aligned[i] - (0.1 * (upper_bb_1d_aligned[i] - lower_bb_1d_aligned[i]))
        
        long_pattern = (body_ratio < 0.3) and near_lower_bb
        short_pattern = (body_ratio < 0.3) and near_upper_bb
        
        # Entry conditions
        if position == 0:
            if long_pattern and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            elif short_pattern and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price crosses back above the middle (SMA) or opposite conditions
            middle_bb_1d = sma_20_1d + std_20_1d  # middle + 0.5*bandwidth
            middle_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, middle_bb_1d.values)
            if close[i] > middle_bb_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price crosses back below the middle (SMA) or opposite conditions
            middle_bb_1d = sma_20_1d - std_20_1d  # middle - 0.5*bandwidth
            middle_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, middle_bb_1d.values)
            if close[i] < middle_bb_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WeeklyTrend_Doji_BB_Pattern"
timeframe = "4h"
leverage = 1.0