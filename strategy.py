#!/usr/bin/env python3
"""
12h_1d_Triple_Barrier_Strategy_v1
Hypothesis: On 12h timeframe, use 1d Donchian(20) breakouts with volume confirmation and 
ATR-based volatility filter to capture medium-term trends. Uses 1d trend filter (price > 1d EMA50 for longs, 
price < 1d EMA50 for shorts) to avoid counter-trend trades. Designed for 15-25 trades/year by requiring
multiple confluence factors: Donchian breakout, volume spike, and trend alignment.
Works in bull markets via breakout longs and in bear markets via breakdown shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Triple_Barrier_Strategy_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian(20), EMA50, and ATR(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper: 20-period high
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower: 20-period low  
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first value
    tr3[0] = tr1[0]  # first value
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high_20_aligned[i]
        breakout_down = close[i] < donch_low_20_aligned[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Volatility filter: ATR > 0.5% of price (avoid choppy markets)
        vol_filter = atr_14_1d_aligned[i] > (close[i] * 0.005)
        
        # Trend filter from 1d EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_spike and vol_filter and above_ema
        short_entry = breakout_down and volume_spike and vol_filter and below_ema
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        long_exit = breakout_down or (close[i] < ema_50_1d_aligned[i])
        short_exit = breakout_up or (close[i] > ema_50_1d_aligned[i])
        
        # Priority: entry > exit > hold
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals