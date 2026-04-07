#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Volume and ADX Filter
# Hypothesis: Donchian channel breakouts capture strong trends. Volume confirms institutional interest.
# ADX > 25 filters for trending markets only. Works in both bull (breakouts up) and bear (breakouts down).
# Target: 15-30 trades/year to minimize fee drag on 12h timeframe.
name = "12h_donchian20_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Donchian Channel (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # ADX(14) for trend strength
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(daily_ema_12h[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below Donchian low or trend weakens (ADX < 20) or turns bearish
            if close[i] < donchian_low[i] or adx[i] < 20 or close[i] < daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price above Donchian high or trend weakens (ADX < 20) or turns bullish
            if close[i] > donchian_high[i] or adx[i] < 20 or close[i] > daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation, strong trend (ADX > 25), and Donchian breakout
            if vol_filter[i] and adx[i] > 25:
                # Long: price breaks above Donchian high + above daily EMA
                if close[i] > donchian_high[i] and close[i] > daily_ema_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low + below daily EMA
                elif close[i] < donchian_low[i] and close[i] < daily_ema_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals