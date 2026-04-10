#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Camarilla levels from 1d: H4 = resistance, L4 = support (breakout levels)
# - Weekly trend filter: price > weekly EMA50 for longs, < weekly EMA50 for shorts
# - Volume confirmation: current 12h volume > 1.8x 30-period average (strict to reduce trades)
# - Entry logic: 
#   * Long: close > H4 AND volume spike AND weekly uptrend
#   * Short: close < L4 AND volume spike AND weekly downtrend
#   * Exit: opposite Camarilla level touch (L4 for long, H4 for short) or ATR trailing stop
# - ATR(20) trailing stop (2.5x) on 12h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivots work well in ranging markets; breakouts capture trends
# - Weekly filter ensures we trade with the higher timeframe trend
# - Strict volume confirmation (1.8x) reduces false breakouts
# - Target: 12-25 trades/year (50-100 total over 4 years) to stay within HARD MAX: 200 total

name = "12h_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
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
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4 = close + range * 1.1/2, L4 = close - range * 1.1/2
    H4 = close_1d + range_1d * 1.1 / 2
    L4 = close_1d - range_1d * 1.1 / 2
    H3 = close_1d + range_1d * 1.1 / 4
    L3 = close_1d - range_1d * 1.1 / 4
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 12h ATR for trailing stop
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_12h = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 12h volume and its 30-period moving average
    volume_12h = prices['volume'].values
    volume_ma_30_12h = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(70, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_12h[i]) or 
            np.isnan(volume_ma_30_12h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 12h volume for filter
        volume_12h_current = volume_12h[i]
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Volume confirmation: current 12h volume > 1.8x 30-period average (strict)
        volume_spike = volume_12h_current > 1.8 * volume_ma_30_12h[i]
        
        # Weekly trend filter
        weekly_uptrend = close_price > ema_50_aligned[i]
        weekly_downtrend = close_price < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: close > H4 AND volume spike AND weekly uptrend
            if close_price > H4_aligned[i] and volume_spike and weekly_uptrend:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else close_price
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short: close < L4 AND volume spike AND weekly downtrend
            elif close_price < L4_aligned[i] and volume_spike and weekly_downtrend:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else close_price
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_12h[i]
                # Exit also if close touches L3 (opposite Camarilla level)
                exit_level = prices['close'].iloc[i] < L3_aligned[i]
                exit_condition = trailing_stop or exit_level
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_12h[i]
                # Exit also if close touches H3 (opposite Camarilla level)
                exit_level = prices['close'].iloc[i] > H3_aligned[i]
                exit_condition = trailing_stop or exit_level
            
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