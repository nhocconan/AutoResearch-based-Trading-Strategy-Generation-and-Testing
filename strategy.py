#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Bull Power = EMA(13, high) - EMA(13, close)
# - Bear Power = EMA(13, close) - EMA(13, low)
# - Long when Bull Power > 0 AND ADX(1d) > 25 (strong trend) AND Bull Power rising
# - Short when Bear Power > 0 AND ADX(1d) > 25 AND Bear Power rising
# - Exit when power fails or ADX < 20 (weak trend)
# - Uses Elder Ray to measure bull/bear strength relative to EMA, ADX to filter ranging markets
# - Target: 12-30 trades/year on 6h timeframe to stay within fee drag limits

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate EMA(13) for Elder Ray
    ema13_high = pd.Series(high).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_low = pd.Series(low).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_close = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = EMA(high) - EMA(close)
    bull_power = ema13_high - ema13_close
    # Bear Power = EMA(close) - EMA(low)
    bear_power = ema13_close - ema13_low
    
    # Calculate 1d ADX for regime filter
    # True Range
    tr1 = df_1d['high'][1:] - df_1d['low'][1:]
    tr2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'][1:] - df_1d['high'][:-1]) > (df_1d['low'][:-1] - df_1d['low'][1:]),
                       np.maximum(df_1d['high'][1:] - df_1d['high'][:-1], 0), 0)
    dm_minus = np.where((df_1d['low'][:-1] - df_1d['low'][1:]) > (df_1d['high'][1:] - df_1d['high'][:-1]),
                        np.maximum(df_1d['low'][:-1] - df_1d['low'][1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF data to LTF
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Track power trends for confirmation
    bull_power_prev = np.roll(bull_power_aligned, 1)
    bear_power_prev = np.roll(bear_power_aligned, 1)
    adx_prev = np.roll(adx_1d_aligned, 1)
    bull_power_prev[0] = np.nan
    bear_power_prev[0] = np.nan
    adx_prev[0] = np.nan
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power_prev[i]) or
            np.isnan(bear_power_prev[i]) or np.isnan(adx_prev[i])):
            signals[i] = 0.0
            continue
        
        # Regime: ADX > 25 = trending market
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
        # Elder Ray signals with trend confirmation
        bull_strong = bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_prev[i]
        bear_strong = bear_power_aligned[i] > 0 and bear_power_aligned[i] > bear_power_prev[i]
        
        if position == 0:  # Flat - look for entry
            if bull_strong and is_trending:
                position = 1
                signals[i] = 0.25
            elif bear_strong and is_trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when bull power fails or trend weakens
            if not bull_strong or is_ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when bear power fails or trend weakens
            if not bear_strong or is_ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals