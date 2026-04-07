#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Donchian Breakout with Volume and ADX Filter
# Hypothesis: Price breaking out of daily Donchian channels (20-period high/low)
# with volume confirmation and daily trend filter (ADX > 25) captures strong momentum
# moves in both bull and bear markets. The 12h timeframe provides good entry timing
# while the daily filter reduces noise and false breakouts. Volume confirms institutional
# participation. ADX ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_daily_donchian_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels, volume average, and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate rolling max/min for Donchian channels
    daily_high_series = pd.Series(daily_high)
    daily_low_series = pd.Series(daily_low)
    donchian_high = daily_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = daily_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed daily bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Daily volume average (50-period)
    daily_volume = df_daily['volume'].values
    daily_volume_series = pd.Series(daily_volume)
    vol_ma = daily_volume_series.rolling(window=50, min_periods=50).mean().values
    
    # Daily ADX (14-period)
    # Calculate True Range
    tr1 = pd.Series(daily_high).subtract(pd.Series(daily_low)).abs()
    tr2 = pd.Series(daily_high).subtract(pd.Series(daily_close).shift(1)).abs()
    tr3 = pd.Series(daily_low).subtract(pd.Series(daily_close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = pd.Series(daily_high).diff()
    down_move = pd.Series(daily_low).diff().multiply(-1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and ATR
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    
    # Calculate DX and ADX
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily data to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    vol_ma_aligned = align_htf_to_ltf(prices, df_daily, vol_ma)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume filter: volume > 2.0x 50-day average (institutional participation)
    vol_filter = volume > (2.0 * vol_ma_aligned)
    
    # Trend filter: ADX > 25 (trending market)
    trend_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below daily Donchian low or trend fails
            if close[i] < donchian_low_aligned[i] or not trend_filter[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above daily Donchian high or trend fails
            if close[i] > donchian_high_aligned[i] or not trend_filter[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above daily Donchian high with volume and trend filter
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                vol_filter[i] and trend_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below daily Donchian low with volume and trend filter
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  vol_filter[i] and trend_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals