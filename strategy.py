#!/usr/bin/env python3
"""
1d_Adaptive_Regime_Breakout_1wTrend_ATRStop_v1
Hypothesis: Daily Donchian(20) breakout filtered by 1-week EMA50 trend and choppiness regime (CHOP>61.8 = range, <38.2 = trend).
In trending regimes (CHOP<38.2): trade breakouts in direction of 1w EMA50.
In ranging regimes (CHOP>61.8): fade breaks of 1d Bollinger Bands (20,2.0) toward SMA20.
Uses ATR(14) stoploss (1.5x) and discrete position sizing (0.25) to minimize fee churn.
Target: 10-20 trades/year per symbol for low fee drag and strong test generalization.
Adaptive regime filter reduces whipsaws in sideways markets while capturing trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA50 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1-week EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) breakout channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Bollinger Bands (20,2.0) for mean reversion in ranging markets
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    
    # Choppiness Index (14) for regime detection
    chop_period = 14
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / np.log10(chop_period) / (highest_high - lowest_low))
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(donchian_high[i]) 
            or np.isnan(donchian_low[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) 
            or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Regime-based entry logic
            if chop[i] < 38.2:  # Trending regime
                # Trade breakouts in direction of 1w EMA50 trend
                long_breakout = price > donchian_high[i]
                long_trend = price > ema_50_1w_aligned[i]
                short_breakout = price < donchian_low[i]
                short_trend = price < ema_50_1w_aligned[i]
                
                if long_breakout and long_trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_breakout and short_trend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:  # Ranging regime (CHOP > 38.2)
                # Fade Bollinger Band extremes toward SMA20
                long_fade = price < bb_lower[i] and price > sma_20[i]
                short_fade = price > bb_upper[i] and price < sma_20[i]
                
                if long_fade:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_fade:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime-based exits
            elif chop[i] < 38.2:  # Trending regime: exit on opposite Donchian break
                if price < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime: exit when price reverts to SMA20
                if price >= sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime-based exits
            elif chop[i] < 38.2:  # Trending regime: exit on opposite Donchian break
                if price > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime: exit when price reverts to SMA20
                if price <= sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Adaptive_Regime_Breakout_1wTrend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0