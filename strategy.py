#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period high/low)
    high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Use previous week's levels to avoid look-ahead
    donchian_upper = np.roll(high_20, 1)
    donchian_lower = np.roll(low_20, 1)
    donchian_upper[0] = np.nan
    donchian_lower[0] = np.nan
    
    # Align weekly Donchian levels to daily timeframe
    upper_daily = align_htf_to_ltf(prices, df_1w, donchian_upper)
    lower_daily = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Daily ADX for trend strength (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr_dm = tr[1:]
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_daily[i]) or np.isnan(lower_daily[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 (strong trend)
        trend_filter = adx[i] > 25
        
        # Long conditions: price breaks above weekly Donchian upper with volume and trend
        long_signal = volume_confirmed and trend_filter and (price_high > upper_daily[i])
        
        # Short conditions: price breaks below weekly Donchian lower with volume and trend
        short_signal = volume_confirmed and trend_filter and (price_low < lower_daily[i])
        
        # Exit when price returns to the opposite Donchian level (mean reversion within the week's range)
        exit_long = position == 1 and price_close < lower_daily[i]
        exit_short = position == -1 and price_close > upper_daily[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily Donchian breakout based on weekly channels with volume confirmation and ADX trend filter.
# Enters long when daily price breaks above the weekly Donchian upper channel (20-week high) with volume >1.5x average and ADX>25.
# Enters short when price breaks below the weekly Donchian lower channel (20-week low) with same conditions.
# Exits when price returns to the opposite Donchian level, capturing mean reversion within the weekly range.
# Weekly timeframe provides structural context, reducing noise from daily fluctuations.
# ADX > 25 ensures trades occur only in strong trends, reducing whipsaws and false breakouts.
# Volume confirmation adds conviction to breakouts.
# Target: 15-25 trades per year to minimize fee drag while capturing significant weekly trends.
# Works in both bull and bear markets by capturing directional moves with proper filtering.