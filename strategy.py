#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H3/L3 breakout + 1d EMA200 trend filter + volume confirmation + session filter (08-20 UTC)
# Camarilla levels provide mean-reversion structure; breakouts beyond H3/L3 indicate strong momentum
# 1d EMA200 ensures we trade with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (1.5x 20-period avg) filters weak breakouts
# Session filter reduces noise during low-liquidity hours
# Works in bull/bear: EMA200 trend filter avoids ranging market failures
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20

name = "4h_1d_camarilla_ema200_volume_session_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (using previous 4h bar's OHLC)
    camarilla_h3 = np.full(len(df_4h), np.nan)
    camarilla_l3 = np.full(len(df_4h), np.nan)
    
    if len(df_4h) >= 1:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        close_4h = df_4h['close'].values
        
        # Camarilla levels: based on previous bar's range
        camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h) / 2
        camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h) / 2
        
        # Align to 1h timeframe (completed 4h bar only)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    else:
        camarilla_h3_aligned = camarilla_l3_aligned = np.full(n, np.nan)
    
    # Load 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend direction
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(avg_volume[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR price < 1d EMA200 (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR price > 1d EMA200 (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla breakout + EMA200 trend filter
            if volume_confirmed:
                # Long entry: price > Camarilla H3 AND price > 1d EMA200 (bullish breakout + uptrend)
                if close[i] > camarilla_h3_aligned[i] and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: price < Camarilla L3 AND price < 1d EMA200 (bearish breakout + downtrend)
                elif close[i] < camarilla_l3_aligned[i] and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals