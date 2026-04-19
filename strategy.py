#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Keltner Channel Breakout with Volume and ADX Trend Filter
# Uses weekly ATR-based channel (EMA20 ± 2*ATR) for breakout signals
# ADX(14) > 25 filters for trending markets only
# Volume > 1.5x 20-day average confirms breakout strength
# Designed for low-frequency, high-conviction trades in both bull and bear markets
# Target: 15-25 trades/year to minimize fee drag
name = "1d_WeeklyKeltnerBreakout_ADX_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Keltner Channel (ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA20 for Keltner middle line
    close_weekly = df_weekly['close'].values
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly ATR for Keltner width
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    tr_weekly = np.maximum(high_weekly - low_weekly, 
                          np.maximum(np.abs(high_weekly - np.roll(close_weekly, 1)), 
                                   np.abs(low_weekly - np.roll(close_weekly, 1))))
    tr_weekly[0] = high_weekly[0] - low_weekly[0]
    atr_weekly = pd.Series(tr_weekly).rolling(window=10, min_periods=10).mean().values
    
    # Keltner Channel bounds: EMA20 ± 2*ATR
    kc_upper_weekly = ema20_weekly + 2.0 * atr_weekly
    kc_lower_weekly = ema20_weekly - 2.0 * atr_weekly
    
    # Align weekly Keltner levels to daily
    kc_upper_aligned = align_htf_to_ltf(prices, df_weekly, kc_upper_weekly)
    kc_lower_aligned = align_htf_to_ltf(prices, df_weekly, kc_lower_weekly)
    ema20_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Daily ADX for trend strength filter
    # Calculate +DI and -DI
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range for ADX
    tr = np.maximum(high - low, 
                   np.maximum(np.abs(high - np.roll(close, 1)), 
                            np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Smoothed +DM, -DM, and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma20[i]
        
        # ADX trend filter: only trade when ADX > 25
        trend_filter = adx[i] > 25
        
        if position == 0:
            # Long: Close breaks above weekly Keltner upper + volume + trend
            if close[i] > kc_upper_aligned[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below weekly Keltner lower + volume + trend
            elif close[i] < kc_lower_aligned[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses below weekly EMA20 or opposite breakout
            if close[i] < ema20_aligned[i] or close[i] < kc_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses above weekly EMA20 or opposite breakout
            if close[i] > ema20_aligned[i] or close[i] > kc_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals