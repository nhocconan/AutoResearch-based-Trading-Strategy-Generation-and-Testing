#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation
# - Donchian channel from 4h: upper/lower bands from 20-period high/low
# - Breakout above upper band (long) or below lower band (short) with volume confirmation
# - 1d EMA200 trend filter ensures we trade with higher timeframe trend (avoids counter-trend in bear markets)
# - Volume confirmation: current volume > 2.0x 20-period average to avoid false breakouts
# - Exit: opposite Donchian band touch (lower band for longs, upper band for shorts)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 20-50 trades/year on 4h (80-200 total over 4 years) to minimize fee drag
# - Works in both bull/bear: EMA200 trend filter adapts to regime, volume confirmation reduces whipsaws

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute 4h Donchian bands (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        # Get current 1d close for trend filter (aligned)
        close_1d_current = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
        
        # 1d trend filter: price > EMA200 = bullish, price < EMA200 = bearish
        bullish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] > trend_aligned[i]
        bearish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > upper Donchian band AND bullish trend AND volume confirmation
            if prices['close'].iloc[i] > donchian_upper[i] and bullish_trend and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: price < lower Donchian band AND bearish trend AND volume confirmation
            elif prices['close'].iloc[i] < donchian_lower[i] and bearish_trend and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price touches opposite Donchian band
            exit_long = prices['close'].iloc[i] < donchian_lower[i]   # Price breaks below lower band (exit long)
            exit_short = prices['close'].iloc[i] > donchian_upper[i]  # Price breaks above upper band (exit short)
            
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