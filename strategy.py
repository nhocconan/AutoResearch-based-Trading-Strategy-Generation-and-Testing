#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d regime filter and volume confirmation
# - Long when 1h closes above Donchian upper(20) AND 4h ADX > 25 (trending) AND 1d volume > 1.3x 20-bar avg
# - Short when 1h closes below Donchian lower(20) AND 4h ADX > 25 (trending) AND 1d volume > 1.3x 20-bar avg
# - Exit when price crosses Donchian midpoint (mean reversion within channel)
# - Uses discrete position sizing (0.20) to minimize fee churn
# - ADX filter ensures we only trade strong trends, avoiding whipsaws in ranging markets
# - Volume confirmation adds conviction to breakouts
# - Session filter (08-20 UTC) reduces noise trades
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in both bull and bear markets: trend following captures directional moves, regime filter avoids false signals

name = "1h_4h_1d_donchian_adx_volume_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h ADX(14) for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    # ADX
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    adx_strong = adx > 25
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.3 * volume_20_avg_1d)
    
    # Align HTF indicators to 1h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_4h, adx_strong)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Donchian Channel(20) on 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper and lower
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Breakout conditions
    breakout_up = close > donchian_upper
    breakout_down = close < donchian_lower
    
    # Exit condition: price crosses midpoint
    exit_long = close < donchian_mid
    exit_short = close > donchian_mid
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_strong_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(breakout_up[i]) or
            np.isnan(breakout_down[i]) or np.isnan(exit_long[i]) or
            np.isnan(exit_short[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND 4h ADX strong AND volume spike
            if (breakout_up[i] and 
                adx_strong_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below Donchian lower AND 4h ADX strong AND volume spike
            elif (breakout_down[i] and 
                  adx_strong_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at midpoint
            # Exit when price crosses Donchian midpoint
            if position == 1:
                if exit_long[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if exit_short[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals