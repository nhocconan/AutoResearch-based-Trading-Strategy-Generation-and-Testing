#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme reversal with 1d trend filter and volume confirmation
    # Williams %R identifies overbought/oversold conditions; 1d EMA50 filters counter-trend trades;
    # volume confirms momentum; discrete sizing minimizes fee drag
    # Target: 12-30 trades/year to avoid overtrading on 6h timeframe
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values if 'volume' in df_6h.columns else np.ones(len(df_6h))
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14) * -100
    
    # Get 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume for confirmation (20-period average)
    vol_avg_20_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Map 6h bar to 1d index for volume confirmation
        idx_1d = i // 4  # 4x 6h bars per day
        if idx_1d >= len(daily_volume):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = daily_volume[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend direction from 1d EMA(50)
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: Williams %R extreme + trend alignment + volume
        # Long: Oversold (< -80) + uptrend + volume
        # Short: Overbought (> -20) + downtrend + volume
        enter_long = (williams_r_aligned[i] < -80) and trend_up and volume_confirmed
        enter_short = (williams_r_aligned[i] > -20) and trend_down and volume_confirmed
        
        # Exit conditions: Williams %R returns to neutral zone (-50)
        exit_long = position == 1 and williams_r_aligned[i] > -50
        exit_short = position == -1 and williams_r_aligned[i] < -50
        
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

name = "6h_1d_williamsr_extreme_trend_volume_v1"
timeframe = "6h"
leverage = 1.0