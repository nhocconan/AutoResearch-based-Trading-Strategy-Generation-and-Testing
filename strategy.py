#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R extreme reversals with 1d EMA200 trend filter
    # Williams %R < -80 = oversold (long), > -20 = overbought (short) on 1d
    # Only trade in direction of 1d EMA200 to avoid counter-trend whipsaws
    # Volume confirmation (>1.3x 24-period average) ensures participation
    # Target: 12-37 trades/year (50-150 total) to minimize fee drag
    # Works in bull/bear markets by only trading with dominant 1d trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_1d = np.full(len(df_1d), np.nan)
    lowest_low_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        highest_high_1d[i] = np.max(high_1d[i-13:i+1])
        lowest_low_1d[i] = np.min(low_1d[i-13:i+1])
    
    williams_r_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        hh = highest_high_1d[i]
        ll = lowest_low_1d[i]
        if hh != ll:  # avoid division by zero
            williams_r_1d[i] = (hh - close_1d[i]) / (hh - ll) * -100
        else:
            williams_r_1d[i] = -50  # neutral when range is zero
    
    # Get 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1d volume for confirmation (>1.3x 24-period average)
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(24, len(df_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-24:i])
    volume_spike_1d = volume_1d > (1.3 * vol_ma_1d)
    
    # Align all indicators to LTF (12h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions
        oversold = williams_r_aligned[i] < -80  # long signal
        overbought = williams_r_aligned[i] > -20  # short signal
        
        # 1d trend filter
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Extreme %R + trend alignment + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike_aligned[i]
        short_entry = overbought and bearish_trend and volume_spike_aligned[i]
        
        # Exit logic: Williams %R returns to neutral range (-50) or trend reversal
        williams_neutral = abs(williams_r_aligned[i] + 50) < 20  # within 20 of -50
        
        long_exit = williams_neutral or not bullish_trend
        short_exit = williams_neutral or not bearish_trend
        
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

name = "12h_1d_williams_r_extreme_ema200_volume_v1"
timeframe = "12h"
leverage = 1.0