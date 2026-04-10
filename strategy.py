#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w trend filter and volume confirmation
# - Williams %R(14) from 1d: oversold < -80 for long, overbought > -20 for short
# - 1w EMA(21) trend filter: only long when price > EMA21, short when price < EMA21
# - Volume confirmation: current 1d volume > 1.3x 20-period MA
# - ATR(14) trailing stop (2.5x) on 6h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Williams %R works well in ranging markets which appear in both bull and bear phases
# - Weekly trend filter prevents counter-trend trades in strong moves
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within HARD MAX: 300 total

name = "6h_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 1d volume and its 20-day moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute 6h ATR for trailing stop
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = np.nan
    tr2_6h[0] = np.nan
    tr3_6h[0] = np.nan
    tr_6h = np.maximum.reduce([tr1_6h, tr2_6h, tr3_6h])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_6h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 1d close for trend filter (use raw close, aligned)
        close_1d_current = close_1d
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
        
        # Get current 1d volume for filter (use raw volume, aligned)
        volume_1d_current = volume_1d
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        
        # Volume confirmation: current 1d volume > 1.3x 20-day average
        volume_confirm = volume_1d_aligned[i] > 1.3 * volume_ma_aligned[i]
        
        # Trend filter: price > weekly EMA21 for long, price < weekly EMA21 for short
        weekly_uptrend = close_1d_aligned[i] > ema_21_aligned[i]
        weekly_downtrend = close_1d_aligned[i] < ema_21_aligned[i]
        
        close_price = close_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold (< -80) AND volume confirmation AND weekly uptrend
            if williams_r_aligned[i] < -80.0 and volume_confirm and weekly_uptrend:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: Williams %R overbought (> -20) AND volume confirmation AND weekly downtrend
            elif williams_r_aligned[i] > -20.0 and volume_confirm and weekly_downtrend:
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
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_6h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_6h[i]
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