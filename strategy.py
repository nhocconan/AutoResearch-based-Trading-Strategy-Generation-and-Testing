#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# - Donchian channels from 1w: upper/lower bands from weekly high/low over 20 periods
# - Breakout above upper band or below lower band with volume confirmation and 1w trend alignment
# - EMA50 trend filter ensures we trade with higher timeframe trend (more responsive than EMA200 for 12h)
# - Volume confirmation: current volume > 1.8x 30-period average to avoid false breakouts
# - Exit: opposite Donchian band touch (lower band for longs, upper band for shorts)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 12-25 trades/year on 12h (50-100 total over 4 years) to minimize fee drag
# - Works in both bull/bear: EMA50 trend filter adapts to regime, volume confirmation reduces whipsaws

name = "12h_1w_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian upper/lower bands
    upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align HTF levels to LTF
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    
    # Pre-compute 12h volume average (30-period)
    volume = prices['volume'].values
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.8x 30-period average
        volume_confirm = volume[i] > 1.8 * vol_ma_30[i]
        
        # Get current 1w close for trend filter (aligned)
        close_1w_current = df_1w['close'].values
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w_current)
        
        # 1w trend filter: price > EMA50 = bullish, price < EMA50 = bearish
        bullish_trend = not np.isnan(close_1w_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1w_aligned[i] > trend_aligned[i]
        bearish_trend = not np.isnan(close_1w_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1w_aligned[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > upper band AND bullish trend AND volume confirmation
            if prices['close'].iloc[i] > upper_aligned[i] and bullish_trend and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: price < lower band AND bearish trend AND volume confirmation
            elif prices['close'].iloc[i] < lower_aligned[i] and bearish_trend and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price touches opposite Donchian band
            exit_long = prices['close'].iloc[i] < lower_aligned[i]   # Price breaks below lower band (exit long)
            exit_short = prices['close'].iloc[i] > upper_aligned[i]  # Price breaks above upper band (exit short)
            
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals