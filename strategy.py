#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 1d trend filter (EMA50/EMA200) and volume spike confirmation.
    # Williams %R identifies overbought/oversold conditions. In strong trends (price > EMA200), 
    # we fade extreme readings expecting mean reversion. Volume spike confirms participation.
    # Target: 50-150 total trades over 4 years = 12-37/year. Works in bull/bear via trend filter.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Williams %R (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d Williams %R (%R = (Highest High - Close) / (Highest High - Lowest Low) * -100)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_1d) / (highest_high - lowest_low)) * -100, -50)
    
    # Align HTF indicators to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h volume spike confirmation (volume > 1.5 * 20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: bullish if close > EMA200, bearish if close < EMA200
        is_bullish_trend = close[i] > ema200_aligned[i]
        is_bearish_trend = close[i] < ema200_aligned[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        # In bullish trend: look for oversold Williams %R for long entries
        # In bearish trend: look for overbought Williams %R for short entries
        if is_bullish_trend and volume_spike[i]:
            long_entry = williams_r_aligned[i] < -80  # Oversold
        elif is_bearish_trend and volume_spike[i]:
            short_entry = williams_r_aligned[i] > -20  # Overbought
        
        # Exit conditions: Williams %R returns to neutral range or opposite extreme
        long_exit = williams_r_aligned[i] > -50  # Return from oversold
        short_exit = williams_r_aligned[i] < -50  # Return from overbought
        
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

name = "6h_1d_williams_r_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0