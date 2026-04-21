#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Only take long when price breaks above Donchian high and 12h EMA50 is rising.
# Only take short when price breaks below Donchian low and 12h EMA50 is falling.
# Volume must be > 1.5x 20-period average for confirmation.
# Exit when price crosses the 12h EMA50 or on opposite Donchian break.
# Target: 20-40 trades/year by requiring strong trend alignment + breakout + volume.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        high_window = prices['high'].iloc[lookback_start:i+1].values
        low_window = prices['low'].iloc[lookback_start:i+1].values
        donchian_high = np.max(high_window)
        donchian_low = np.min(low_window)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # 12h EMA50 trend: rising if current > previous, falling if current < previous
        ema_now = ema_50_aligned[i]
        ema_prev = ema_50_aligned[i-1]
        ema_rising = ema_now > ema_prev
        ema_falling = ema_now < ema_prev
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above Donchian high + EMA50 rising
                if price > donchian_high and ema_rising:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low + EMA50 falling
                elif price < donchian_low and ema_falling:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below EMA50 or breaks below Donchian low
                if price < ema_now or price < donchian_low:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above EMA50 or breaks above Donchian high
                if price > ema_now or price > donchian_high:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0