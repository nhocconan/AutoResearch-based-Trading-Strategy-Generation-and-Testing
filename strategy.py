#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX(14) regime filter and volume confirmation
- Donchian breakout captures momentum in trending markets
- 1d ADX > 25 confirms strong trend (avoids whipsaws in ranging markets)
- Volume > 1.5x 20-period average ensures breakout has participation
- Only trade in direction of 1d EMA(50) for higher probability
- Designed for 4h timeframe targeting 30-60 trades/year (120-240 over 4 years)
- Works in bull markets (long breakouts) and bear markets (short breakouts)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend direction
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
        return adx.values
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_donchian_high = close[i] > donchian_high[i]
        price_below_donchian_low = close[i] < donchian_low[i]
        
        # Trend and regime filters
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        strong_trend = adx_aligned[i] > 25
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long conditions: breakout above Donchian high, uptrend, strong trend, volume
            long_signal = (price_above_donchian_high and 
                          uptrend and
                          strong_trend and
                          volume_spike)
            
            # Short conditions: breakout below Donchian low, downtrend, strong trend, volume
            short_signal = (price_below_donchian_low and 
                           downtrend and
                           strong_trend and
                           volume_spike)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or trend weakness
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below Donchian low or trend weakens
                if (price_below_donchian_low or 
                    not uptrend or
                    adx_aligned[i] < 20):  # Trend weakening
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above Donchian high or trend weakens
                if (price_above_donchian_high or 
                    not downtrend or
                    adx_aligned[i] < 20):  # Trend weakening
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dADX14_Regime_EMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0