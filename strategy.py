# 72637
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1-day EMA(34) trend filter and volume confirmation.
# In up-trend (price > 1d EMA34): long when price breaks above Donchian upper band.
# In down-trend (price < 1d EMA34): short when price breaks below Donchian lower band.
# Volume must exceed 1.5x 20-period average for confirmation. Exit on opposite band touch.
# Target: 30-60 trades/year by requiring trend alignment + breakout + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1. Load 1-day HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1-day EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 2. Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
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
        
        # Trend filter: price vs 1-day EMA34
        price_above_ema = price > ema_34_1d_aligned[i]
        price_below_ema = price < ema_34_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Up-trend: look for long breakout
                if price_above_ema and price > donchian_high:
                    signals[i] = 0.25
                    position = 1
                # Down-trend: look for short breakdown
                elif price_below_ema and price < donchian_low:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit long when price touches or breaks below Donchian lower band
                if price <= donchian_low:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit short when price touches or breaks above Donchian upper band
                if price >= donchian_high:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_1dEMA34Trend_Volume"
timeframe = "4h"
leverage = 1.0