#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R extreme + 1d EMA200 trend filter + volume confirmation
    # Williams %R identifies overbought/oversold conditions (long when < -80, short when > -20)
    # EMA200 on 1d defines the major trend (only long when price > EMA200, short when < EMA200)
    # Volume spike (>2.0x 20-period average) confirms institutional participation
    # Target: 12-30 trades/year (48-120 total over 4 years) to minimize fee drag
    # Works in both bull (trend continuation) and bear (mean reversion from extremes) markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close_1d[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate EMA200 on 1d
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 12h volume for confirmation (>2.0x 20-period average)
    vol_ma_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_12h[i] = np.mean(volume[i-20:i])
    volume_spike_12h = volume > (2.0 * vol_ma_12h)
    
    # Align all indicators to LTF (12h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Precompute session filter (08-20 UTC) - optional for 12h but kept for consistency
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_spike_12h[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80  # extreme oversold
        overbought = williams_r_aligned[i] > -20  # extreme overbought
        
        # 1d EMA200 trend filter
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Extreme Williams %R + trend alignment + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike_12h[i]
        short_entry = overbought and bearish_trend and volume_spike_12h[i]
        
        # Exit logic: Williams %R returns to neutral territory (-50) or trend reversal
        long_exit = williams_r_aligned[i] > -50 or not bullish_trend
        short_exit = williams_r_aligned[i] < -50 or not bearish_trend
        
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

name = "12h_1d_williamsr_extreme_ema200_volume_v1"
timeframe = "12h"
leverage = 1.0