#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Donchian breakout: price > upper channel (20-period high) or < lower channel (20-period low)
# - 1d EMA(50) trend filter: ensures trading with higher timeframe trend direction
# - 12h volume > 1.8x 20-period average confirms breakout strength
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14)
# - Discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in bull/bear via trend filter + volatility-adjusted stops

name = "12h_1d_donchian_volume_trend_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian(20) channels
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Upper channel: 20-period high
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h volume confirmation: > 1.8x 20-period average
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.8 * avg_volume_20_12h)
    
    # 12h ATR(14) for stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_12h = np.zeros_like(tr)
    atr_14_12h[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_12h[i] = (atr_14_12h[i-1] * (14-1) + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_spike_12h[i]) or 
            np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks below lower Donchian
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                prices['close'].iloc[i] < donchian_lower[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks above upper Donchian
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                prices['close'].iloc[i] > donchian_upper[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike_12h[i]:
                # Long signal: price breaks above upper Donchian in 1d uptrend
                if prices['close'].iloc[i] > donchian_upper[i] and prices['close'].iloc[i] > ema_50_1d_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = 0.25
                # Short signal: price breaks below lower Donchian in 1d downtrend
                elif prices['close'].iloc[i] < donchian_lower[i] and prices['close'].iloc[i] < ema_50_1d_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = -0.25
    
    return signals