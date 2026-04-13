# 12h_1d_1w_Camarilla_Pivot_Breakout_Volume
# Hypothesis: Uses Camarilla pivot levels from daily timeframe (1d) on 12h chart with volume confirmation and weekly trend filter.
# Trades breakouts above R4 or below S4 with volume > 1.5x 20-period average.
# Weekly trend filter ensures we only trade in direction of weekly EMA50 to avoid counter-trend whipsaws.
# Works in both bull and bear markets by capturing volatility expansion after consolidation near key levels.
# Target: 12-37 trades/year (50-150 total over 4 years) on 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels for each day
    # Using previous day's OHLC to calculate today's levels
    camarilla_r4 = np.full_like(close_1d, np.nan)
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_r2 = np.full_like(close_1d, np.nan)
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    camarilla_s2 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    camarilla_s4 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        range_val = ph - pl
        if range_val <= 0:
            continue
            
        camarilla_r4[i] = pc + range_val * 1.1 / 2
        camarilla_r3[i] = pc + range_val * 1.1 / 4
        camarilla_r2[i] = pc + range_val * 1.1 / 6
        camarilla_r1[i] = pc + range_val * 1.1 / 12
        camarilla_s1[i] = pc - range_val * 1.1 / 12
        camarilla_s2[i] = pc - range_val * 1.1 / 6
        camarilla_s3[i] = pc - range_val * 1.1 / 4
        camarilla_s4[i] = pc - range_val * 1.1 / 2
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all signals to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(camarilla_r4_aligned[i]) or \
           np.isnan(camarilla_s4_aligned[i]) or \
           np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R4 with volume expansion and weekly uptrend
        if close[i] > camarilla_r4_aligned[i] and volume_expansion[i] and close[i] > ema_50_1w_aligned[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short condition: price breaks below S4 with volume expansion and weekly downtrend
        elif close[i] < camarilla_s4_aligned[i] and volume_expansion[i] and close[i] < ema_50_1w_aligned[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Hold current position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        # Flat
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0