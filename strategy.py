#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1-week Bollinger Band squeeze breakout + volume confirmation + ADX trend filter.
# Uses Bollinger Band width < 20th percentile for squeeze detection, breakout on close > upper band or < lower band,
# volume > 1.5x 20-day average, and ADX > 25 for trend confirmation. Designed for low trade frequency (~10-20/year)
# to minimize fee decay while capturing explosive moves in both bull and bear markets.

name = "1d_1w_bb_squeeze_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on 1w data
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 20-period SMA and standard deviation
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std()
    
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_width = upper_band - lower_band
    
    # Bollinger Band width percentile (20-period lookback for squeeze)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True)
    
    # Squeeze condition: BB width < 20th percentile
    squeeze = bb_width_percentile < 0.2
    
    # Breakout conditions: close outside Bollinger Bands
    breakout_up = close_1w > upper_band
    breakout_down = close_1w < lower_band
    
    # ADX calculation on 1w data (14-period)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_14 / tr_14)
    minus_di = 100 * (minus_dm_14 / tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Align all 1w indicators to daily timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze.values)
    breakout_up_aligned = align_htf_to_ltf(prices, df_1w, breakout_up.values)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1w, breakout_down.values)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx.values.fillna(0))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 50 to ensure all indicators are valid
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(squeeze_aligned[i]) or np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: BB squeeze breakout with volume and trend confirmation
        long_entry = (squeeze_aligned[i] and breakout_up_aligned[i] and 
                      volume_filter[i] and (adx_aligned[i] > 25))
        short_entry = (squeeze_aligned[i] and breakout_down_aligned[i] and 
                       volume_filter[i] and (adx_aligned[i] > 25))
        
        # Exit conditions: return to middle Bollinger Band (20-period SMA)
        sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20.values.fillna(0))
        if not np.isnan(sma_20_aligned[i]):
            exit_long = close[i] < sma_20_aligned[i]
            exit_short = close[i] > sma_20_aligned[i]
        else:
            exit_long = exit_short = False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals