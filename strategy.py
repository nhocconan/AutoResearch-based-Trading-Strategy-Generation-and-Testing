#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX regime filter
    # Long when: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND ADX > 25 (trending)
    # Short when: Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND ADX > 25 (trending)
    # Exit when: Opposite Elder Ray signal OR ADX < 20 (range regime)
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Elder Ray measures bull/bear power via EMA13; ADX filters for trending markets only.
    # Works in bull (trend continuation) and bear (strong downtrends) by requiring ADX>25.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Elder Ray and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(13) for Elder Ray
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_12h = high_12h - ema_13_12h
    bear_power_12h = low_12h - ema_13_12h
    
    # Calculate 12h ADX(14)
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # +DM = max(H-Hprev, 0) if H-Hprev > Lprev-L else 0
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    # -DM = max(Lprev-L, 0) if Lprev-L > H-Hprev else 0
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha=1/period)
    def wilder_smooth(values, period):
        smoothed = np.full_like(values, np.nan)
        if len(values) < period:
            return smoothed
        # First value: simple average
        smoothed[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(values)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        return smoothed
    
    atr_12h = wilder_smooth(tr, 14)
    plus_di_12h = 100 * wilder_smooth(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilder_smooth(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilder_smooth(dx_12h, 14)
    
    # Align HTF indicators to 6h timeframe (wait for completed 12h bar)
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # warmup for indicators
        # Skip if data not ready
        if (np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions
        bullish_momentum = bull_power_12h_aligned[i] > 0 and bear_power_12h_aligned[i] < 0
        bearish_momentum = bear_power_12h_aligned[i] < 0 and bull_power_12h_aligned[i] > 0
        
        # ADX regime filter: ADX > 25 = trending, ADX < 20 = range
        trending = adx_12h_aligned[i] > 25
        ranging = adx_12h_aligned[i] < 20
        
        # Entry conditions
        long_entry = bullish_momentum and trending and position != 1
        short_entry = bearish_momentum and trending and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (not bullish_momentum or ranging))
        exit_short = (position == -1 and (not bearish_momentum or ranging))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0