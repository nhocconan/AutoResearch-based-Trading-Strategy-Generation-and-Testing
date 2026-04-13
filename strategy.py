#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 12h EMA trend filter
    # Works in both bull and bear: Elder Ray measures bull/bear power via EMA13,
    # 12h EMA20 provides trend alignment, volume confirmation ensures momentum.
    # Discrete sizing (0.25) minimizes fee drag. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 13-period EMA for Elder Ray (using 1d close)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Calculate 20-period EMA for 12h trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 1d volume for confirmation (20-period average)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema20_12h_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        idx_1d = i // 4  # 6h bars per day = 4
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price relative to 12h EMA20
        price_above_ema = close[i] > ema20_12h_aligned[i]
        price_below_ema = close[i] < ema20_12h_aligned[i]
        
        # Entry conditions: Elder Ray extremes + trend alignment + volume
        # Long: strong bull power (> 0.75 * std) + uptrend + volume
        # Short: strong bear power (< -0.75 * std) + downtrend + volume
        bull_std = np.nanstd(bull_power[max(0, idx_1d-20):idx_1d+1]) if idx_1d > 0 else 1.0
        bear_std = np.nanstd(bear_power[max(0, idx_1d-20):idx_1d+1]) if idx_1d > 0 else 1.0
        
        # Avoid division by zero in std calculation
        if bull_std == 0:
            bull_std = 1.0
        if bear_std == 0:
            bear_std = 1.0
            
        bull_threshold = 0.75 * bull_std
        bear_threshold = -0.75 * bear_std
        
        enter_long = (bull_power_aligned[i] > bull_threshold) and price_above_ema and volume_confirmed
        enter_short = (bear_power_aligned[i] < bear_threshold) and price_below_ema and volume_confirmed
        
        # Stoploss: 1.5x ATR based on 1d true range
        if idx_1d < len(high_1d) and idx_1d < len(low_1d) and idx_1d < len(close_1d):
            tr1 = high_1d[idx_1d] - low_1d[idx_1d]
            tr2 = abs(high_1d[idx_1d] - close_1d[idx_1d-1]) if idx_1d > 0 else 0
            tr3 = abs(low_1d[idx_1d] - close_1d[idx_1d-1]) if idx_1d > 0 else 0
            true_range = max(tr1, tr2, tr3)
            atr_1d = true_range  # Simplified: using current bar's true range
        else:
            atr_1d = 0
        stop_distance = atr_1d * 1.5
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1d_12h_elder_ray_power_ema_volume_v1"
timeframe = "6h"
leverage = 1.0