#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation
# - Donchian(20) breakout on 4h: long when close > upper band, short when close < lower band
# - ATR filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid low volatility chop
# - Volume confirmation: current 4h volume > 1.3x 20-period average
# - Weekly trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - ATR(14) trailing stop (2.5x) on 4h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within HARD MAX: 400 total

name = "4h_1w_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_1d = np.maximum.reduce([tr1, tr2, tr3])
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Calculate rolling max/min for Donchian channels
    upper_channel = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR for trailing stop
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr1_4h[0] = np.nan
    tr2_4h[0] = np.nan
    tr3_4h[0] = np.nan
    tr_4h = np.maximum.reduce([tr1_4h, tr2_4h, tr3_4h])
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 4h volume and its 20-period moving average
    volume_4h = prices['volume'].values
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(volume_ma_20_4h[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current values
        close_price = close_4h[i]
        volume_4h_current = volume_4h[i]
        
        # ATR filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid low volatility chop
        atr_filter = atr_14_aligned[i] > 0.5 * atr_50_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirm = volume_4h_current > 1.3 * volume_ma_20_4h[i]
        
        # Weekly trend filter
        weekly_uptrend = close_4h[i] > ema_50_aligned[i]
        weekly_downtrend = close_4h[i] < ema_50_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close_price > upper_channel[i]
        short_breakout = close_price < lower_channel[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Donchian breakout + ATR filter + volume confirm + weekly uptrend
            if long_breakout and atr_filter and volume_confirm and weekly_uptrend:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short entry: Donchian breakout + ATR filter + volume confirm + weekly downtrend
            elif short_breakout and atr_filter and volume_confirm and weekly_downtrend:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_4h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_4h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals