#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
    # Williams %R < -80 = oversold (long), > -20 = overbought (short)
    # Only trade in direction of 12h EMA34 to avoid counter-trend whipsaws
    # Volume spike (>1.5x 20-period average) confirms participation
    # Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
    # Works in bull/bear markets by only trading with the dominant 12h trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Williams %R(14) on 12h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    # Replace division by zero with NaN
    williams_r[highest_high == lowest_low] = np.nan
    
    # Get 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    volume_spike_12h = volume_12h > (1.5 * vol_ma_12h)
    
    # Align all indicators to LTF (6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # 12h trend filter
        bullish_trend = close[i] > ema34_12h_aligned[i]
        bearish_trend = close[i] < ema34_12h_aligned[i]
        
        # Entry logic: Williams %R extreme + trend alignment + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike_aligned[i]
        short_entry = overbought and bearish_trend and volume_spike_aligned[i]
        
        # Exit logic: Williams %R returns to neutral zone (-50) or trend change
        williams_r_neutral = abs(williams_r_aligned[i] + 50) < 30  # Within 30 points of -50
        trend_change = (position == 1 and not bullish_trend) or (position == -1 and not bearish_trend)
        
        long_exit = williams_r_neutral or trend_change
        short_exit = williams_r_neutral or trend_change
        
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

name = "6h_12h_williams_r_ema34_volume_v1"
timeframe = "6h"
leverage = 1.0