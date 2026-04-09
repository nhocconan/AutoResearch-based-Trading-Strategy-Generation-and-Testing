#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation
# - Uses 12h Donchian(20) breakout for entry signals (structure)
# - Confirms with 1w EMA(50) trend filter (only trade in direction of weekly trend)
# - Requires 12h volume > 1.8x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 2.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to manage drawdown in bear markets
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)
# - Donchian breakouts work in trending markets; weekly EMA filter avoids counter-trend trades in bear markets
# - Volume filter reduces false breakouts, ATR stop manages risk

name = "12h_1w_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Volume > 1.8x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_12h[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i]) or atr_12h[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and trend filter
            if (high[i] >= donchian_high[i] and    # Break above upper Donchian
                close[i] > ema_50_1w_aligned[i] and   # Weekly uptrend filter
                volume_spike[i]):                     # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            elif (low[i] <= donchian_low[i] and    # Break below lower Donchian
                  close[i] < ema_50_1w_aligned[i] and  # Weekly downtrend filter
                  volume_spike[i]):                    # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals