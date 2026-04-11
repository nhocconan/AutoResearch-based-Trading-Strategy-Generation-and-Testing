#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Long: Price breaks above 20-period 12h Donchian high + 1d close > 1d EMA50 + volume > 1.5x 20-period 12h average
# - Short: Price breaks below 20-period 12h Donchian low + 1d close < 1d EMA50 + volume > 1.5x 20-period 12h average
# - Exit: Opposite Donchian breakout or trend reversal
# - Williams Alligator identifies trend direction and alignment
# - Elder Ray measures bull/bear power behind the move
# - Volume confirmation filters out weak signals
# - Works in both bull (strong bull power) and bear (strong bear power) markets
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "12h_donchian_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return signals
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 12h Donchian channels (20-period)
    # Donchian high: highest high over 20 periods
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over 20 periods
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        high_current = high[i]
        low_current = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_high = close_current > donch_high[i-1]  # Break above previous period's high
        breakout_low = close_current < donch_low[i-1]    # Break below previous period's low
        
        # 1d trend filter
        uptrend_1d = close_current > ema50_1d_aligned[i]
        downtrend_1d = close_current < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout high + 1d uptrend + volume confirmation
        if breakout_high and uptrend_1d and vol_confirm:
            enter_long = True
        
        # Short: Donchian breakout low + 1d downtrend + volume confirmation
        if breakout_low and downtrend_1d and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below Donchian low OR 1d trend turns down
            exit_long = (close_current < donch_low[i]) or (not uptrend_1d)
        elif position == -1:
            # Exit short if price breaks above Donchian high OR 1d trend turns up
            exit_short = (close_current > donch_high[i]) or (not downtrend_1d)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals