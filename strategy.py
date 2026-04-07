#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4H Weekly Donchian Breakout with Volume and ADX Filter
# Hypothesis: Breakouts from weekly Donchian channels (20-period) with volume confirmation
# and ADX trend strength filter capture strong momentum moves. Weekly timeframe provides
# more robust levels than daily, reducing false breakouts. Works in both bull and bear
# markets by trading breakouts in the direction of the weekly trend. Target: 20-40 trades/year.

name = "4h_weekly_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and ADX
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly high/low/close for calculations
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    donchian_high = weekly_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed weekly bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Calculate ADX (14-period) for trend strength
    # True Range
    tr1 = pd.Series(weekly_high).subtract(pd.Series(weekly_low)).abs()
    tr2 = pd.Series(weekly_high).subtract(pd.Series(weekly_close).shift(1)).abs()
    tr3 = pd.Series(weekly_low).subtract(pd.Series(weekly_close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(weekly_high).diff()
    down_move = pd.Series(weekly_low).diff().abs()  # positive when down
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr
    
    # DX and ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Handle NaN values
    adx = adx.fillna(0).values
    
    # Align weekly data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Volume filter: volume > 1.5x 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly Donchian low or weak trend (ADX < 20)
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above weekly Donchian high or weak trend (ADX < 20)
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Strong trend filter: ADX > 25
            if adx_aligned[i] > 25:
                # Long entry: breakout above weekly Donchian high with volume
                if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: breakdown below weekly Donchian low with volume
                elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals