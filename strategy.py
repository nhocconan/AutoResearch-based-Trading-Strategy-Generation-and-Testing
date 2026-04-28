#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA34 trend + volume spike + ATR(14) stoploss
# Donchian breakouts capture momentum; 1w EMA34 ensures alignment with weekly trend.
# Volume spike (>2x 20-bar average) confirms breakout strength.
# ATR-based trailing stop limits drawdown in ranging/bear markets.
# Works in both bull/bear by requiring trend alignment and using discrete position sizing (0.25).
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for stoploss and position sizing (using 1d HTF for stability)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period: no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # Approximation for first bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # Approximation for first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Donchian(20) channels on 12h chart (using 1d HTF for structure, aligned to 12h)
    # Donchian uses 20-period high/low of the 1d timeframe, aligned to 12h
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Volume confirmation: >2.0x 20-bar average volume (on 12h chart)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 34, 20)  # Donchian(20), EMA(34), ATR(14), Volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA trend filter
        ema_trend_up = close[i] > ema_34_1w_aligned[i]
        ema_trend_down = close[i] < ema_34_1w_aligned[i]
        
        price = close[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest
            if price < highest_since_entry - 2.5 * atr_14_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            # Exit on Donchian lower band break (failed breakout)
            elif price < donch_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest
            if price > lowest_since_entry + 2.5 * atr_14_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            # Exit on Donchian upper band break (failed breakdown)
            elif price > donch_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat - look for new entries
            # Long entry: Price > Donchian upper band, 1w EMA uptrend, volume confirm
            if price > donch_high_20_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short entry: Price < Donchian lower band, 1w EMA downtrend, volume confirm
            elif price < donch_low_20_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
            else:
                signals[i] = 0.0
    
    return signals