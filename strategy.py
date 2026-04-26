#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: 12h Camarilla R3/S3 breakout in direction of 1d EMA34 trend, confirmed by volume spike (>2x 20-bar MA) and choppiness regime (CHOP > 61.8 = ranging, fade breakouts; CHOP < 38.2 = trending, follow breakouts). Uses ATR(14) trailing stop (2.0 ATR from extreme). Designed for low frequency (target 12-37 trades/year) to avoid fee drag, works in bull/bear via trend alignment and regime filter. Discrete position sizing (0.25) minimizes churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Fractals for Camarilla levels (need 2-bar confirmation delay)
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Bullish fractal = potential resistance (sell fractal), Bearish fractal = potential support (buy fractal)
    # Camarilla R3/S3 derived from prior day's range
    # We'll compute Camarilla levels using prior 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's range for Camarilla
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    range_1d = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + range_1d * 1.1 / 4
    camarilla_s3 = prev_close_1d - range_1d * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3, additional_delay_bars=0)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3, additional_delay_bars=0)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR(14) for stoploss calculation
    atr_period = 14
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Choppiness Index regime filter (14-period)
    chop_period = 14
    sum_tr = np.zeros(n)
    if n >= chop_period:
        sum_tr[chop_period-1] = np.sum(tr[:chop_period])
    for i in range(chop_period, n):
        sum_tr[i] = sum_tr[i-1] - tr[i-chop_period] + tr[i]
    
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(n):
        if i < chop_period:
            max_high[i] = np.max(high[:i+1]) if i >= 0 else 0
            min_low[i] = np.min(low[:i+1]) if i >= 0 else 0
        else:
            max_high[i] = np.max(high[i-chop_period+1:i+1])
            min_low[i] = np.min(low[i-chop_period+1:i+1])
    
    chop = np.zeros(n)
    for i in range(chop_period-1, n):
        if sum_tr[i] > 0 and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (max_high[i] - min_low[i])) / np.log10(chop_period)
        else:
            chop[i] = 50.0  # neutral
    
    # Regime: CHOP > 61.8 = ranging (fade), CHOP < 38.2 = trending (follow)
    chop_high = 61.8
    chop_low = 38.2
    ranging = chop > chop_high
    trending = chop < chop_low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Track extreme for trailing stop
    long_high = 0.0
    low_low = 0.0
    
    # Warmup: max of calculations
    start_idx = max(34, 20, 20, atr_period, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(chop[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        chop_val = chop[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume spike
        # In trending regime: follow breakout direction
        # In ranging regime: fade breakout (opposite direction)
        long_entry = False
        short_entry = False
        
        if trending[i]:
            # Trending: follow breakout
            long_entry = (close_val > r3_val) and bullish_1d and vol_spike
            short_entry = (close_val < s3_val) and bearish_1d and vol_spike
        elif ranging[i]:
            # Ranging: fade breakout
            long_entry = (close_val < s3_val) and bearish_1d and vol_spike  # fade downside breakout
            short_entry = (close_val > r3_val) and bullish_1d and vol_spike  # fade upside breakout
        
        # Update trailing extremes
        if position == 1:
            long_high = max(long_high, high_val)
        elif position == -1:
            low_low = min(low_low, low_val)
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # ATR trailing stop or opposite Camarilla level
            if long_high > 0 and close_val < (long_high - 2.0 * atr_val):
                exit_long = True
            elif close_val < s3_val:  # Opposite breakout (S3)
                exit_long = True
        elif position == -1:
            # ATR trailing stop or opposite Camarilla level
            if low_low > 0 and close_val > (low_low + 2.0 * atr_val):
                exit_short = True
            elif close_val > r3_val:  # Opposite breakout (R3)
                exit_short = True
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            long_high = high_val  # Reset extreme on new entry
            low_low = 0.0
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            low_low = low_val  # Reset extreme on new entry
            long_high = 0.0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            long_high = 0.0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            low_low = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0