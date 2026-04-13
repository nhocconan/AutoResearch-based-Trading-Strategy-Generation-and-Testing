#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h strategy using 1d Williams %R extremes with 1d EMA trend filter and volume confirmation
    # Williams %R captures short-term reversals (<-80 oversold, >-20 overbought)
    # EMA filter ensures trend alignment (price > EMA for longs, < EMA for shorts)
    # Volume confirmation filters low-momentum signals
    # Discrete sizing (0.25) minimizes fee drag. Target: 12-30 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA (21-period)
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Get 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get 1d index for volume confirmation
        idx_1d = i // 2  # 2x 12h bars per 1d
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d[idx_1d] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price relative to 1d EMA
        price_above_ema = close[i] > ema_21_1d_aligned[i]
        price_below_ema = close[i] < ema_21_1d_aligned[i]
        
        # Entry conditions: Williams %R extremes + trend alignment + volume
        enter_long = (williams_r_aligned[i] < -80) and price_above_ema and volume_confirmed
        enter_short = (williams_r_aligned[i] > -20) and price_below_ema and volume_confirmed
        
        # Stoploss: ATR-based using 1d range
        if idx_1d < len(high_1d) and idx_1d < len(low_1d):
            daily_range = high_1d[idx_1d] - low_1d[idx_1d]
            stop_distance = daily_range * 0.4  # 40% of daily range
        else:
            stop_distance = 0
        
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

name = "12h_1d_williams_r_extreme_ema_volume_v1"
timeframe = "12h"
leverage = 1.0