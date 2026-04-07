#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + 1-day EMA50 up + volume > 1.5x avg
# Short when price breaks below Donchian(20) low + 1-day EMA50 down + volume > 1.5x avg
# Exit when price crosses Donchian midpoint or trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day EMA50
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().shift(1).values
    ema50_up = ema50_1d > ema50_1d_prev
    ema50_down = ema50_1d < ema50_1d_prev
    
    # Align EMA trend to 4h
    ema50_up_aligned = align_htf_to_ltf(prices, df_1d, ema50_up)
    ema50_down_aligned = align_htf_to_ltf(prices, df_1d, ema50_down)
    
    # 4-period Donchian channels (20)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # 4-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_up_aligned[i]) or np.isnan(ema50_down_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or trend reverses
            elif close[i] < donchian_mid[i] or ema50_down_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or trend reverses
            elif close[i] > donchian_mid[i] or ema50_up_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation and trend filter
            vol_surge = volume[i] > 1.5 * vol_avg[i]
            
            # Long: break above Donchian high + uptrend + volume surge
            if close[i] > high_max[i] and ema50_up_aligned[i] and vol_surge:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low + downtrend + volume surge
            elif close[i] < low_min[i] and ema50_down_aligned[i] and vol_surge:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals