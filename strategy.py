#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA34 trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets (price > EMA34),
# we look for pullbacks to oversold levels (-80) for long entries and overbought levels (-20) for shorts.
# Volume confirmation ensures institutional participation. Designed for 12h timeframe to reduce
# trade frequency and avoid fee drag. Works in both bull and bear markets by following the trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for EMA34 (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on daily close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R on 12h data (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema_34_aligned[i]
        wr = williams_r[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Only trade in direction of trend (price > EMA34 = uptrend, price < EMA34 = downtrend)
            if price > ema_trend:  # Uptrend
                # Look for oversold pullback to enter long
                if wr <= -80 and vol_spike:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Look for overbought bounce to enter short
                if wr >= -20 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of trend
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when overbought or trend turns down
                if wr >= -20 or price < ema_trend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when oversold or trend turns up
                if wr <= -80 or price > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_EMA34_Volume"
timeframe = "12h"
leverage = 1.0