#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation
# - Donchian(20) from 4h: breakout above upper band = long, below lower band = short
# - 1d EMA200 to filter trades in direction of higher timeframe trend (works in bull/bear)
# - Volume confirmation: current 4h volume > 2.0x 20-period average (strict to reduce trades)
# - ATR-based trailing stop: exit when price moves against position by 2.5*ATR
# - Designed for 4h timeframe: targets 20-50 trades/year to avoid fee drag
# - Uses discrete position sizing (0.25) to minimize fee churn
# - EMA200 on 1d provides strong trend filter that works across market regimes

name = "4h_1d_donchian_ema200_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Pre-compute 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper band: highest high over past 20 periods
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low over past 20 periods
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (2.0 * avg_volume_20)  # Strict threshold to reduce trades
    
    # Pre-compute 4h ATR(14) for trailing stop
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0  # for trailing stop
    lowest_low = 0.0    # for trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            if close_4h[i] > highest_high:
                highest_high = close_4h[i]
            # Exit: trailing stop hit
            if close_4h[i] < highest_high - 2.5 * atr_14[i]:
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            if close_4h[i] < lowest_low:
                lowest_low = close_4h[i]
            # Exit: trailing stop hit
            if close_4h[i] > lowest_low + 2.5 * atr_14[i]:
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Breakout long: price closes above upper Donchian band AND above 1d EMA200
                if close_4h[i] > highest_20[i] and close_4h[i] > ema_200_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    highest_high = close_4h[i]
                    signals[i] = 0.25
                # Breakout short: price closes below lower Donchian band AND below 1d EMA200
                elif close_4h[i] < lowest_20[i] and close_4h[i] < ema_200_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    lowest_low = close_4h[i]
                    signals[i] = -0.25
    
    return signals