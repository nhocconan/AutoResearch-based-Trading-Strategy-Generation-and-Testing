#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation.
# In bull/bear markets: follow breakouts in direction of 1d EMA trend.
# Avoids counter-trend trades. Uses volume > 1.5x 20-period average for confirmation.
# Target: 12-30 trades/year by requiring trend alignment + breakout + volume confirmation.
# Works in bull (follows uptrend breaks) and bear (follows downtrend breaks).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
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
        
        # Trend filter: price vs 1d EMA(34)
        price_above_ema = price > ema_34_1d_aligned[i]
        price_below_ema = price < ema_34_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Follow breakouts in direction of 1d trend
                if price > donchian_high and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                elif price < donchian_low and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: opposite breakout or trend reversal
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian low OR trend reversal
                if price < donchian_low or price < ema_34_1d_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above Donchian high OR trend reversal
                if price > donchian_high or price > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0