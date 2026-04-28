#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA200 trend filter and ATR(14) volatility filter.
# Uses 4h primary timeframe targeting 19-50 trades/year (75-200 total over 4 years).
# 12h EMA200 provides strong trend filter: bull when price > EMA200, bear when price < EMA200.
# Donchian(20) breakout captures institutional price levels with proven edge.
# ATR(14) < 0.02*price ensures low volatility environment to avoid choppy whipsaws.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.

name = "4h_Donchian20_12hEMA200_Trend_ATR_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Donchian channels and 12h data for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 20 or len(df_12h) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA200 for trend filter
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) for volatility filter on 1d
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure sufficient history for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_200_12h_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) < 2% of price (avoid choppy markets)
        vol_filter = atr_14_aligned[i] < 0.02 * close[i]
        
        # Trend filter: 12h EMA200 direction (price above/below EMA200)
        price_above_ema = close[i] > ema_200_12h_aligned[i]
        price_below_ema = close[i] < ema_200_12h_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        long_entry = price_above_ema and long_breakout and vol_filter
        short_entry = price_below_ema and short_breakout and vol_filter
        
        # Exit conditions: opposite Donchian level (mean reversion)
        long_exit = close[i] < donchian_low_aligned[i]  # Exit long at lower band
        short_exit = close[i] > donchian_high_aligned[i]  # Exit short at upper band
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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