#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Donchian Breakout with Volume and ADX Filter
# Hypothesis: Weekly Donchian channels (20-period) capture major trends. 
# Price breaking above weekly upper band with volume and ADX>25 indicates strong momentum continuation.
# Price breaking below weekly lower band with volume and ADX>25 indicates strong downtrend continuation.
# Works in both bull and bear markets: In bull, breaks above upper band continue up; breaks below lower band get bought (mean reversion in strong uptrend).
# In bear, breaks below lower band continue down; breaks above upper band get sold (mean reversion in strong downtrend).
# Volume filter ensures institutional participation. ADX filter ensures trending market.
# Target: 15-25 trades/year (60-100 over 4 years).

name = "1d_weekly_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate rolling max/min for Donchian channels
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    weekly_upper = weekly_high_series.rolling(window=20, min_periods=20).max().values
    weekly_lower = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous week's completed data (avoid look-ahead)
    weekly_upper_prev = np.roll(weekly_upper, 1)
    weekly_lower_prev = np.roll(weekly_lower, 1)
    weekly_upper_prev[0] = weekly_upper_prev[1] if len(weekly_upper_prev) > 1 else 0
    weekly_lower_prev[0] = weekly_lower_prev[1] if len(weekly_lower_prev) > 1 else 0
    
    # Align to daily timeframe (use previous week's levels)
    weekly_upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper_prev)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower_prev)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ADX filter: ADX > 25 for trending market
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # Align ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back to weekly lower band or ADX weakens
            if (close[i] <= weekly_lower_aligned[i] or adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back to weekly upper band or ADX weakens
            if (close[i] >= weekly_upper_aligned[i] or adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above weekly upper band with volume and strong trend
            if (high[i] > weekly_upper_aligned[i] and 
                close[i] > weekly_upper_aligned[i] and 
                vol_filter[i] and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly lower band with volume and strong trend
            elif (low[i] < weekly_lower_aligned[i] and 
                  close[i] < weekly_lower_aligned[i] and 
                  vol_filter[i] and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals