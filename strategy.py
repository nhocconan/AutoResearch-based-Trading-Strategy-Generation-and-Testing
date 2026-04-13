#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h strategy using 1h RSI extremes with 12h EMA trend filter and volume confirmation
    # Works in both bull and bear: RSI < 30 for long, > 70 for short captures mean reversion,
    # 12h EMA > price for longs, < price for shorts ensures trend alignment,
    # volume confirmation ensures momentum. Discrete sizing (0.25) minimizes fee drag.
    # Target: 20-40 trades/year to stay within 4h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1h data for RSI (primary HTF for reversal signals)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Get 1h volume for confirmation (20-period average)
    volume_1h = df_1h['volume'].values if 'volume' in df_1h.columns else np.ones(len(df_1h))
    
    # Calculate 1h RSI (14-period)
    delta = pd.Series(close_1h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate 12h EMA (50-period)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1h volume average (20-period)
    vol_avg_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1h, rsi)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_avg_20_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_avg_20_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_avg_20_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        idx_1h = i // 3  # 1h bars in 4h timeframe (3 bars per 4h)
        if idx_1h >= len(volume_1h):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1h[idx_1h] > 1.5 * vol_avg_20_1h_aligned[i]
        
        # Trend filter: price relative to 12h EMA
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions: RSI extremes + trend alignment + volume
        enter_long = (rsi_aligned[i] < 30) and price_above_ema and volume_confirmed
        enter_short = (rsi_aligned[i] > 70) and price_below_ema and volume_confirmed
        
        # Stoploss: 2x ATR based on 1h true range (simplified using hourly range)
        hourly_range = high_1h[idx_1h] - low_1h[idx_1h] if idx_1h < len(high_1h) else 0
        stop_distance = hourly_range * 1.5  # 150% of hourly range
        
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

name = "4h_1h_12h_rsi_extreme_ema_volume_v1"
timeframe = "4h"
leverage = 1.0