#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Donchian channel breakout with 12h EMA34 trend filter and volume confirmation.
# Enter long when price breaks above weekly Donchian upper (20) with volume spike and above 12h EMA34.
# Enter short when price breaks below weekly Donchian lower (20) with volume spike and below 12h EMA34.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 12-37 trades/year.
# Weekly structure provides major trend context, volume confirms breakout validity, EMA34 filters intermediate trend on 6h.
# Works in bull (breakouts with trend) and bear (failed breaks reverse) markets.

name = "6h_WeeklyDonchian20_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    n_1w = len(high_1w)
    donchian_high_20 = np.full(n_1w, np.nan)
    donchian_low_20 = np.full(n_1w, np.nan)
    
    for i in range(20, n_1w):
        # Use rolling window of previous 20 weekly bars (including current)
        donchian_high_20[i] = np.max(high_1w[i-20:i])
        donchian_low_20[i] = np.min(low_1w[i-20:i])
    
    # Forward fill Donchian levels
    donchian_high_20 = pd.Series(donchian_high_20).ffill().values
    donchian_low_20 = pd.Series(donchian_low_20).ffill().values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly and 12h indicators to 6h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA34
        above_ema = close[i] > ema_34_12h_aligned[i]
        below_ema = close[i] < ema_34_12h_aligned[i]
        
        # Weekly Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > donchian_high_20_aligned[i] and volume_spike[i]
        short_breakout = close[i] < donchian_low_20_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or trend reversal
        long_exit = close[i] < donchian_low_20_aligned[i] or below_ema
        short_exit = close[i] > donchian_high_20_aligned[i] or above_ema
        
        # Handle entries and exits
        if long_breakout and above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and below_ema and position >= 0:
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