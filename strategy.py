#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation.
# Uses Camarilla levels from 1d pivots for stronger structure with 4h EMA34 for trend.
# Long when price breaks above R3 with volume and price > 4h EMA34 (uptrend).
# Short when price breaks below S3 with volume and price < 4h EMA34 (downtrend).
# Volume spike (>2.0x 12-bar average) confirms breakout strength.
# Session filter: 08-20 UTC to reduce noise trades outside active market hours.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 80-120 total trades over 4 years = 20-30/year for 12h.

name = "12h_Camarilla_R3S3_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Get 1d data for pivot calculation (stronger HTF structure)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1d Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d_prev) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels (R3/S3 provide good breakout structure)
    R3 = pivot + range_1d * 1.1 / 4.0
    S3 = pivot - range_1d * 1.1 / 4.0
    
    # Align to 12h timeframe (use previous 1d bar's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate 12h volume spike: >2.0x 12-bar average volume (more conservative)
    volume_series = pd.Series(volume)
    volume_ma_12 = volume_series.rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > 2.0 * volume_ma_12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for EMA34 and pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(volume_ma_12[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_4h_aligned[i]
        price_below_ema = close[i] < ema_34_4h_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < S3_aligned[i] or close[i] < ema_34_4h_aligned[i]
        short_exit = close[i] > R3_aligned[i] or close[i] > ema_34_4h_aligned[i]
        
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