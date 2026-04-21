#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# In bull markets: long when price breaks above Donchian high and above 1d EMA34.
# In bear markets: short when price breaks below Donchian low and below 1d EMA34.
# Volume > 1.5x 20-period average confirms breakout strength.
# Uses discrete position sizing (0.30) to limit trades and reduce fee drag.
# Target: 25-40 trades/year by requiring trend alignment + breakout + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
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
        
        # Trend filter: price vs 1d EMA34
        trend_up = price > ema_34_aligned[i]
        trend_down = price < ema_34_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long breakout: price above Donchian high and above 1d EMA34
                if price > donchian_high and trend_up:
                    signals[i] = 0.30
                    position = 1
                # Short breakout: price below Donchian low and below 1d EMA34
                elif price < donchian_low and trend_down:
                    signals[i] = -0.30
                    position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of trend/volume
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian low or loss of uptrend
                if price < donchian_low or not trend_up:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above Donchian high or loss of downtrend
                if price > donchian_high or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_DonchianBreakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0