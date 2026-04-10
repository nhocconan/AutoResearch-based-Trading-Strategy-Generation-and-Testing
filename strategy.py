#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA50) and volume confirmation
# - Long when price breaks above 6h Donchian upper channel (20-bar high) in weekly uptrend with volume spike
# - Short when price breaks below 6h Donchian lower channel (20-bar low) in weekly downtrend with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based trailing stop: exit when price moves 2.0x ATR against position from extreme point
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Weekly trend filter reduces whipsaws in bear markets; volume confirmation ensures institutional participation

name = "6h_1w_donchian_breakout_volume_trend_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute weekly indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h Donchian channels (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # 6h ATR(14) for trailing stop
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    close_6h = prices['close'].values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_6h = np.zeros_like(tr)
    atr_14_6h[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_6h[i] = (atr_14_6h[i-1] * (14-1) + tr[i]) / 14
    
    # 6h volume confirmation: > 1.5x 20-period average
    volume_6h = prices['volume'].values
    avg_volume_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike_6h = volume_6h > (1.5 * avg_volume_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    extreme_price = 0.0  # Tracks highest high for long, lowest low for short
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_14_6h[i]) or np.isnan(vol_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update extreme price (highest high since entry)
            if prices['high'].iloc[i] > extreme_price:
                extreme_price = prices['high'].iloc[i]
            
            # Exit: 2.0x ATR trailing stop from extreme price
            if prices['close'].iloc[i] < extreme_price - 2.0 * atr_14_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update extreme price (lowest low since entry)
            if prices['low'].iloc[i] < extreme_price:
                extreme_price = prices['low'].iloc[i]
            
            # Exit: 2.0x ATR trailing stop from extreme price
            if prices['close'].iloc[i] > extreme_price + 2.0 * atr_14_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with weekly trend and volume filters
            if vol_spike_6h[i]:
                # Long signal: price breaks above Donchian high in weekly uptrend
                if (prices['close'].iloc[i] > donchian_high[i] and 
                    prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_6h[i]
                    extreme_price = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low in weekly downtrend
                elif (prices['close'].iloc[i] < donchian_low[i] and 
                      prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_6h[i]
                    extreme_price = prices['low'].iloc[i]
                    signals[i] = -0.25
    
    return signals