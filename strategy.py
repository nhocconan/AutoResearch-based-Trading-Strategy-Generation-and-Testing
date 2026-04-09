#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakouts with volume confirmation and ATR trailing stop
# 4h Donchian(20) provides clear trend structure, proven across market regimes
# Volume confirmation (current 1h volume > 1.8x 20-period average) filters false breakouts
# ATR trailing stop (2.0x ATR) manages risk and adapts to volatility
# Session filter (08-20 UTC) reduces noise trades
# Target: 60-150 total trades over 4 years = 15-37/year for 1h
# Works in bull/bear: price reacts to 4h structure, volume confirms validity, ATR stop controls drawdown

name = "1h_4h_donchian_volume_atr_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian channels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    
    # Pre-compute ATR(14) for 1h timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.8x average 1h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # ATR trailing stop: exit if price drops 2.0x ATR from highest
            if close[i] < highest_since_long - 2.0 * atr[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # ATR trailing stop: exit if price rises 2.0x ATR from lowest
            if close[i] > lowest_since_short + 2.0 * atr[i]:
                position = 0
                lowest_since_short = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Breakout trading with volume confirmation
            # Long on Donchian high breakout, Short on Donchian low breakout
            if volume_confirmed:
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    highest_since_long = close[i]
                    signals[i] = 0.20
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    lowest_since_short = close[i]
                    signals[i] = -0.20
    
    return signals