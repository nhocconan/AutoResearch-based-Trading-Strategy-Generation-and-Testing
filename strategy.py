#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeS
# Hypothesis: Uses 4h Camarilla pivot levels with 1d trend filter (EMA34) and volume spike for 1h entries.
# Trades only during 08-20 UTC to avoid low-liquidity hours. Targets 15-30 trades/year by requiring
# confluence of 4h structure, 1d trend, and volume confirmation. Designed to work in both bull and bear
# markets by filtering trades with higher-timeframe trend and avoiding choppy periods via volume spike.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeS"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    camarilla_range = high_4h - low_4h
    r1_4h = close_4h + 1.1 * camarilla_range / 12
    s1_4h = close_4h - 1.1 * camarilla_range / 12
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_1h = align_htf_to_ltf(prices, df_4h, s1_4h)
    ema_34_1d_1h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter on 1h (24-period average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or 
            np.isnan(ema_34_1d_1h[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > R1, above 1d EMA34 trend, volume spike
            if close[i] > r1_1h[i] and close[i] > ema_34_1d_1h[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
                bars_since_entry = 0
            # Short: Price < S1, below 1d EMA34 trend, volume spike
            elif close[i] < s1_1h[i] and close[i] < ema_34_1d_1h[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit conditions: require minimum 4 bars held to reduce whipsaw
            if bars_since_entry >= 4:
                if close[i] < r1_1h[i] or close[i] < ema_34_1d_1h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            else:
                # Hold position for minimum period
                signals[i] = 0.20
        elif position == -1:
            # Exit conditions: require minimum 4 bars held
            if bars_since_entry >= 4:
                if close[i] > s1_1h[i] or close[i] > ema_34_1d_1h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
            else:
                # Hold position for minimum period
                signals[i] = -0.20
    
    return signals