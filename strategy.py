#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Weekly Camarilla levels provide stronger structural support/resistance than daily.
# Long when price breaks above weekly R4 with volume and price > 1d EMA34 (uptrend).
# Short when price breaks below weekly S4 with volume and price < 1d EMA34 (downtrend).
# Volume spike (>2.0x 24-bar average) confirms breakout strength.
# Session filter: 08-20 UTC to reduce noise trades outside active market hours.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 75-150 total trades over 4 years = 19-37/year for 6h.

name = "6h_Camarilla_R4S4_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid per-bar datetime conversion
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get weekly data for Camarilla calculation (stronger HTF structure)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly Camarilla levels from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_prev = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w_prev) / 3.0
    # Range = H - L
    range_1w = high_1w - low_1w
    # Camarilla levels (R4/S4 provide strong breakout/breakdown structure)
    R4 = pivot + range_1w * 1.1 / 2.0
    S4 = pivot - range_1w * 1.1 / 2.0
    
    # Align to 6h timeframe (use previous weekly bar's levels)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Calculate 6h volume spike: >2.0x 24-bar average volume (more conservative)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for EMA34 and pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(volume_ma_24[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Weekly Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R4_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S4_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < S4_aligned[i] or close[i] < ema_34_1d_aligned[i]
        short_exit = close[i] > R4_aligned[i] or close[i] > ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals