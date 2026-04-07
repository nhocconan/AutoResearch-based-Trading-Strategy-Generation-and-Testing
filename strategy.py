#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with 1-week trend filter and volume confirmation
# Long when price breaks above 20-day high + weekly EMA(21) is rising + volume > 1.5x 20-day average
# Short when price breaks below 20-day low + weekly EMA(21) is falling + volume > 1.5x 20-day average
# Exit when price crosses 10-day EMA in opposite direction or weekly EMA flips
# Stoploss at 2 * ATR(14)
# Position size: 0.30 (30% of capital)
# Designed for low trade frequency to minimize fee drag while capturing major trends
# Works in both bull and bear markets via short/long symmetry

name = "1d_donchian20_1w_ema_vol_v1"
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
    
    # 1-week data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21)
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_21 = close_1w_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_prev = np.roll(ema_21, 1)
    ema_21_prev[0] = ema_21[0]
    ema_rising = ema_21 > ema_21_prev
    ema_falling = ema_21 < ema_21_prev
    
    # Align weekly EMA signals to daily
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    # 1-day Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1-day volume average (20-period)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # 1-day ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1-day EMA(10) for exit
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_rising_aligned[i]) or 
            np.isnan(ema_falling_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_10[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 10-day EMA or weekly EMA turns falling
            elif close[i] < ema_10[i] or ema_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 10-day EMA or weekly EMA turns rising
            elif close[i] > ema_10[i] or ema_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for breakouts with volume confirmation and weekly trend filter
            volume_surge = volume[i] > 1.5 * vol_ma[i]
            
            # Long: price breaks above Donchian high + volume surge + weekly EMA rising
            if close[i] > donchian_high[i] and volume_surge and ema_rising_aligned[i]:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume surge + weekly EMA falling
            elif close[i] < donchian_low[i] and volume_surge and ema_falling_aligned[i]:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals