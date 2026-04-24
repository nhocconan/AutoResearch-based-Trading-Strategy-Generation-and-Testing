#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ADX regime filter + volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ADX regime filter.
- Donchian breakout: Long when price > highest high of last 20 periods, Short when price < lowest low.
- Regime filter: ADX(14) > 25 = trending (trade breakouts), ADX < 20 = ranging (fade breakouts).
- Volume confirmation: current volume > 1.5x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend, and fading breakouts in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_di_smooth / atr
    minus_di = 100 * minus_di_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels on 12h
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 34, 20)  # Donchian(20) + ADX buffer + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime: ADX > 25 = trending, ADX < 20 = ranging
            if adx_aligned[i] > 25:
                # Trending regime: trade breakouts in trend direction
                # Use 1d EMA34 for trend direction
                ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
                ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
                if not np.isnan(ema_34_1d_aligned[i]) and not np.isnan(ema_34_1d_aligned[i-1]):
                    ema34_slope = ema_34_1d_aligned[i] - ema_34_1d_aligned[i-1]
                    if close[i] > highest_high[i] and ema34_slope > 0 and volume_spike[i]:
                        # Uptrend: buy on upside breakout
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < lowest_low[i] and ema34_slope < 0 and volume_spike[i]:
                        # Downtrend: sell on downside breakdown
                        signals[i] = -0.25
                        position = -1
            elif adx_aligned[i] < 20:
                # Ranging regime: fade breakouts (mean reversion)
                if close[i] > highest_high[i] and volume_spike[i]:
                    # Price broke above Donchian high in range: sell expecting reversion
                    signals[i] = -0.25
                    position = -1
                elif close[i] < lowest_low[i] and volume_spike[i]:
                    # Price broke below Donchian low in range: buy expecting reversion
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Long exit: price returns to midline or opposite breakout
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midline or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midline or opposite breakout
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midline or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX_Regime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0