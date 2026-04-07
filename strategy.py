#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter (1w EMA50) and daily volume confirmation
# Long when price breaks above Donchian(20) high, weekly EMA50 > previous weekly EMA50 (uptrend), and daily volume > 1.5x 10-period average
# Short when price breaks below Donchian(20) low, weekly EMA50 < previous weekly EMA50 (downtrend), and daily volume > 1.5x 10-period average
# Exit when price crosses Donchian midpoint or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly EMA50 for trend filter and daily volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) - within proven range for 12h timeframe

name = "12h_donchian20_1w_ema50_vol_v1"
timeframe = "12h"
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
    
    # 12h Donchian(20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Weekly data for trend filter (1w EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_prev = np.roll(ema50_1w, 1)
    ema50_1w_prev[0] = ema50_1w[0]  # handle first value
    ema50_1w_rising = ema50_1w > ema50_1w_prev  # uptrend when rising
    ema50_1w_falling = ema50_1w < ema50_1w_prev  # downtrend when falling
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_rising)
    ema50_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_falling)
    
    # Daily volume for confirmation (10-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema50_1w_rising_aligned[i]) or np.isnan(ema50_1w_falling_aligned[i]) or
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr[i])):
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
            # Exit: price crosses Donchian midpoint or trend changes to down
            elif close[i] < donch_mid[i] or ema50_1w_falling_aligned[i]:
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
            # Exit: price crosses Donchian midpoint or trend changes to up
            elif close[i] > donch_mid[i] or ema50_1w_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, trend alignment, and volume confirmation
            # Bullish breakout: price breaks above Donchian(20) high
            bullish_break = close[i] > donch_high[i] and close[i-1] <= donch_high[i-1]
            # Bearish breakout: price breaks below Donchian(20) low
            bearish_break = close[i] < donch_low[i] and close[i-1] >= donch_low[i-1]
            
            # Volume confirmation: current daily volume > 1.5x 10-day average
            volume_confirmed = volume[i] > 1.5 * volume_ma_1d_aligned[i]
            
            # Long: bullish breakout, weekly uptrend, volume confirmation
            if (bullish_break and
                ema50_1w_rising_aligned[i] and
                volume_confirmed):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, weekly downtrend, volume confirmation
            elif (bearish_break and
                  ema50_1w_falling_aligned[i] and
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals