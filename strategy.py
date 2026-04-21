#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA200 filter and volume confirmation.
# Williams %R measures momentum on a scale of -100 to 0. Readings below -80 indicate oversold,
# above -20 indicate overbought. In trending markets, we can ride extremes; in ranging markets,
# we fade extremes. We use 1d EMA200 to determine trend: only take long signals when price > EMA200
# in uptrend, short when price < EMA200 in downtrend. Volume confirmation ensures legitimacy.
# Works in bull markets (riding momentum) and bear markets (fading bounces in downtrend).
# Target: 20-40 trades/year by requiring Williams %R extremes, EMA trend filter, and volume spike.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA200 and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA200 on daily timeframe
    close_d = df_1d['close'].values
    ema200_d = pd.Series(close_d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 14-period Williams %R on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily EMA200 to 6h (wait for daily close)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        vol_current = volume[i]
        
        # Trend filter: price relative to daily EMA200
        above_ema = price_close > ema200_1d_aligned[i]
        below_ema = price_close < ema200_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_current > 1.5 * vol_ma_20[i]
        
        # Williams %R levels
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        if position == 0:
            # Enter long when oversold, above EMA200 (uptrend), and volume confirmation
            if oversold and above_ema and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short when overbought, below EMA200 (downtrend), and volume confirmation
            elif overbought and below_ema and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to neutral (-50) or price crosses below EMA200
                if williams_r[i] >= -50:
                    exit_signal = True
                elif not above_ema:  # price crossed below EMA200
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R returns to neutral (-50) or price crosses above EMA200
                if williams_r[i] <= -50:
                    exit_signal = True
                elif not below_ema:  # price crossed above EMA200
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA200_Volume"
timeframe = "6h"
leverage = 1.0