#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
# Enter long when price breaks above 20-bar Donchian high with volume > 2.0x 50-bar average and close > 1d EMA50.
# Enter short when price breaks below 20-bar Donchian low with volume > 2.0x average and close < 1d EMA50.
# Exit when price returns to the 20-bar Donchian midpoint.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
# Works in bull markets (breakouts continue up with trend) and bear markets (breakdowns continue down with trend).
# Uses 4h Donchian for structure (proven edge) and 1d EMA50 for trend filter (reduces whipsaws).

name = "4h_Donchian_20_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (MTF trend)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate volume confirmation: >2.0x 50-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_50 = volume_series.rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > 2.0 * volume_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA50 bias
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Exit condition: return to midpoint
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
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