#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(20) pullback in 4h trend direction with volume confirmation and session filter (08-20 UTC).
# Uses 4h EMA50 for trend filter, 1h EMA20 for entry timing on pullbacks to the 4h trend.
# Volume spike (>1.5x 20-bar average) confirms momentum. Session filter avoids low-liquidity hours.
# Designed for 1h timeframe to capture medium-term swings with controlled trade frequency (~20-50/year).
# Works in bull markets by buying pullbacks to rising 4h EMA50, and in bear markets by selling rallies to falling 4h EMA50.

name = "1h_EMA20_Pullback_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h EMA20 for pullback entries
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # EMA50 needs 50 bars, EMA20 needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_20_1h[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA50 direction
        price_above_4h_ema = close[i] > ema_50_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_50_4h_aligned[i]
        
        # Pullback to 1h EMA20 conditions
        long_pullback = low[i] <= ema_20_1h[i]  # Price touches or goes below EMA20
        short_pullback = high[i] >= ema_20_1h[i]  # Price touches or goes above EMA20
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_4h_ema and long_pullback and vol_confirm
        short_entry = price_below_4h_ema and short_pullback and vol_confirm
        
        # Exit conditions: opposite pullback with volume
        long_exit = short_pullback and vol_confirm  # Rally to EMA20 with volume = exit long
        short_exit = long_pullback and vol_confirm  # Pullback to EMA20 with volume = exit short
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals