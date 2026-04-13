#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
    # Works in bull/bear: Donchian captures breakouts, 1w Supertrend filters trend direction,
    # volume confirms momentum, ATR stop controls risk. Target: 20-50 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for Supertrend (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(10) for Supertrend
    tr1w = np.maximum(np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1])), np.abs(low_1w[1:] - close_1w[:-1]))
    tr1w = np.concatenate([[np.nan], tr1w])
    atr1w = pd.Series(tr1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1w Supertrend (period=10, multiplier=3.0)
    hl2_1w = (high_1w + low_1w) / 2
    upperband_1w = hl2_1w + (3.0 * atr1w)
    lowerband_1w = hl2_1w - (3.0 * atr1w)
    
    supertrend_1w = np.full_like(close_1w, np.nan)
    direction_1w = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        if np.isnan(atr1w[i]) or atr1w[i] == 0:
            supertrend_1w[i] = np.nan
            direction_1w[i] = np.nan
            continue
            
        if close_1w[i] > upperband_1w[i-1]:
            direction_1w[i] = 1
        elif close_1w[i] < lowerband_1w[i-1]:
            direction_1w[i] = -1
        else:
            direction_1w[i] = direction_1w[i-1]
            if direction_1w[i] == 1 and lowerband_1w[i] < lowerband_1w[i-1]:
                lowerband_1w[i] = lowerband_1w[i-1]
            if direction_1w[i] == -1 and upperband_1w[i] > upperband_1w[i-1]:
                upperband_1w[i] = upperband_1w[i-1]
        
        if direction_1w[i] == 1:
            supertrend_1w[i] = lowerband_1w[i]
        else:
            supertrend_1w[i] = upperband_1w[i]
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 1d primary timeframe
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    # Calculate 1d ATR for stoploss
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(supertrend_1w_aligned[i]) or 
            np.isnan(direction_1w_aligned[i]) or
            np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(atr_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period average
        volume_confirmed = volume_1d[i] > 1.8 * vol_avg_20_1d_aligned[i]
        
        # Trend direction from 1w Supertrend
        trend_up = direction_1w_aligned[i] == 1
        trend_down = direction_1w_aligned[i] == -1
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_20_aligned[i]
        breakout_down = close[i] < donchian_low_20_aligned[i]
        
        # Entry conditions
        enter_long = trend_up and volume_confirmed and breakout_up
        enter_short = trend_down and volume_confirmed and breakout_down
        
        # Stoploss conditions
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - atr_multiplier * atr_1d[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + atr_multiplier * atr_1d[i]
        
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

name = "1d_1w_donchian_breakout_volume_atrstop_v1"
timeframe = "1d"
leverage = 1.0