#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper band AND 1d close > 1d EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below 1d Donchian lower band AND 1d close < 1d EMA50 AND volume > 2.0x 20-period average
# Exit when price crosses 1d EMA50 (trend reversal) OR price retouches the 1d Donchian middle band (mean reversion)
# Uses 12h primary timeframe with 1d HTF for all indicators (Donchian bands, EMA50)
# Donchian breakouts capture strong momentum moves, EMA50 filters counter-trend noise
# Volume confirmation ensures breakouts have conviction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Donchian20_Breakout_1dEMA50_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for all indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Donchian channels (20-period) based on previous day's data (no look-ahead)
    if len(df_1d) >= 20:
        # Use rolling window on previous day's data to avoid look-ahead
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Calculate Donchian upper/lower bands based on previous 20 days
        donchian_upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
        
        # Align to 12h timeframe
        donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
        donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
        donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    else:
        donchian_upper_aligned = np.full(n, np.nan)
        donchian_lower_aligned = np.full(n, np.nan)
        donchian_middle_aligned = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND 1d close > 1d EMA50 AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND 1d close < 1d EMA50 AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend reversal) OR price retouches Donchian middle band (mean reversion)
            if close[i] < ema_50_1d_aligned[i] or abs(close[i] - donchian_middle_aligned[i]) < 0.001 * donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend reversal) OR price retouches Donchian middle band (mean reversion)
            if close[i] > ema_50_1d_aligned[i] or abs(close[i] - donchian_middle_aligned[i]) < 0.001 * donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals