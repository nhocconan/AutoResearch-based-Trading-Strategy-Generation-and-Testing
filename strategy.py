#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams %R extreme + 1w EMA50 trend filter + volume confirmation
    # Williams %R(14) < -80 = oversold (long), > -20 = overbought (short)
    # 1w EMA50 ensures we trade with the weekly trend to avoid counter-trend whipsaws
    # Volume > 1.5x 20-period average confirms participation
    # Session filter (08-20 UTC) reduces low-liquidity noise
    # Target: 15-25 trades/year (60-100 total) to minimize fee drag
    # Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Get 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume for confirmation (>1.5x 20-period average)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_1d)
    
    # Align all indicators to LTF (1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # 1w trend filter
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Entry logic: Extreme %R + trend alignment + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike[i]
        short_entry = overbought and bearish_trend and volume_spike[i]
        
        # Exit logic: %R returns to neutral zone (-50) or trend reversal
        williams_r_neutral = abs(williams_r_aligned[i] + 50) < 25  # Within 25 points of -50
        long_exit = williams_r_neutral or not bullish_trend
        short_exit = williams_r_neutral or not bearish_trend
        
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

name = "1d_1w_williamsr_extreme_ema50_volume_v1"
timeframe = "1d"
leverage = 1.0