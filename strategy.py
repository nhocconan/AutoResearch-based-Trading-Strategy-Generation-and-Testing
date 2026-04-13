#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
    # Long: BB width < 20th percentile (squeeze) + price breaks above upper BB + ADX > 20 (trending) + volume > 1.5x avg
    # Short: BB width < 20th percentile (squeeze) + price breaks below lower BB + ADX > 20 (trending) + volume > 1.5x avg
    # Exit: price returns to middle BB (20-period SMA)
    # Uses 6h primary timeframe for lower trade frequency vs 4h, suitable for 6h timeframe constraints
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    # BB squeeze breakouts work in both ranging and trending markets when confirmed with trend and volume
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values if 'volume' in df_6h.columns else np.ones(len(df_6h))
    
    # Get 1d data for ADX and volume (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate Bollinger Bands on 6h data (20-period, 2 std dev)
    sma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20 * 100  # percentage width
    
    # Calculate BB width percentile (20-period lookback for regime)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # Calculate ADX on 1d data (14-period)
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:]))
    tr = np.concatenate([[np.nan], tr1])
    
    # +DM and -DM
    dm_plus = np.where((high_1d[1:] - np.roll(high_1d, 1)[1:]) > (np.roll(low_1d, 1)[1:] - low_1d[1:]),
                       np.maximum(high_1d[1:] - np.roll(high_1d, 1)[1:], 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1)[1:] - low_1d[1:]) > (high_1d[1:] - np.roll(high_1d, 1)[1:]),
                        np.maximum(np.roll(low_1d, 1)[1:] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume averages
    vol_avg_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    sma_20_aligned = align_htf_to_ltf(prices, df_6h, sma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_6h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_6h, lower_bb)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_6h, bb_width_percentile)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_avg_20_6h)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(sma_20_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_avg_20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width below 20th percentile (low volatility)
        is_squeeze = bb_width_percentile_aligned[i] < 20
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume_6h[i] > 1.5 * vol_avg_20_6h_aligned[i]
        
        # ADX filter: trending market (ADX > 20)
        is_trending = adx_aligned[i] > 20
        
        # Breakout conditions
        breakout_up = close_6h[i] > upper_bb_aligned[i]
        breakout_down = close_6h[i] < lower_bb_aligned[i]
        
        # Entry conditions
        enter_long = is_squeeze and breakout_up and volume_confirmed and is_trending
        enter_short = is_squeeze and breakout_down and volume_confirmed and is_trending
        
        # Exit conditions: price returns to middle BB (20-period SMA)
        exit_long = position == 1 and close_6h[i] <= sma_20_aligned[i]
        exit_short = position == -1 and close_6h[i] >= sma_20_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_bb_squeeze_breakout_v1"
timeframe = "6h"
leverage = 1.0