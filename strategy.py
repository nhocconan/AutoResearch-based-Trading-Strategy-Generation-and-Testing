#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Daily Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Uses weekly EMA34 to determine long-term trend (bull/bear) and daily Donchian(20) for entry.
# In weekly uptrend: buy breakouts above daily Donchian high with volume confirmation
# In weekly downtrend: sell breakdowns below daily Donchian low with volume confirmation
# Designed to capture major trends while avoiding counter-trend whipsaws.
# Targets 12-30 trades/year with disciplined risk control to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load daily data for Donchian channels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (20-period)
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate 20-period average daily volume for volume spike detection
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        weekly_ema = ema_34_1w_aligned[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        vol_ma = vol_ma_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Determine weekly trend
            is_uptrend = price > weekly_ema
            is_downtrend = price < weekly_ema
            
            if is_uptrend:
                # Weekly uptrend: look for long breakouts
                if price > upper and vol_spike:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend:
                # Weekly downtrend: look for short breakdowns
                if price < lower and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below daily Donchian low
                if price < lower:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above daily Donchian high
                if price > upper:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyEMA34_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0