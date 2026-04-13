#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme reversal with 12h EMA50 trend filter and volume confirmation
    # Williams %R < -80 = oversold (long), > -20 = overbought (short)
    # Only trade with 12h EMA50 trend to avoid counter-trend whipsaws
    # Volume > 1.5x 20-period average confirms institutional participation
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R (14-period) on 6h
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    
    # Get 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_6h[i] = np.mean(volume[i-20:i])
    volume_spike_6h = volume > (1.5 * vol_ma_6h)
    
    # Align all indicators to LTF (6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # 12h trend filter
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Entry logic: Extreme %R + trend alignment + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike_6h[i]
        short_entry = overbought and bearish_trend and volume_spike_6h[i]
        
        # Exit logic: %R returns to neutral zone (-50) or trend reversal
        neutral_zone = abs(williams_r_aligned[i] + 50) < 10  # Within 10 points of -50
        trend_reversal = (position == 1 and not bullish_trend) or (position == -1 and not bearish_trend)
        
        long_exit = neutral_zone or trend_reversal
        short_exit = neutral_zone or trend_reversal
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_williams_r_extreme_ema50_volume_v1"
timeframe = "6h"
leverage = 1.0