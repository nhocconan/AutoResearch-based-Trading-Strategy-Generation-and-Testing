#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and 1d volume confirmation
# Long when price breaks above 6h Donchian upper (20) + weekly pivot bias bullish (close > weekly pivot) + 1d volume > 1.3x 20-period avg
# Short when price breaks below 6h Donchian lower (20) + weekly pivot bias bearish (close < weekly pivot) + 1d volume > 1.3x 20-period avg
# Weekly pivot bias provides structural filter to avoid counter-trend trades in ranging markets
# Discrete position sizing (0.25) to control drawdown. Target: 50-150 total trades over 4 years

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data once before loop for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1w data once before loop for pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 6h Indicator: Donchian Channels (20-period) ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (already aligned, but ensure proper shifting)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # === 1w Indicator: Weekly Pivot Point for bias ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly pivot bias: bullish if close > pivot, bearish if close < pivot
        weekly_close_1w = close_1w[-1] if len(close_1w) > 0 else 0  # latest weekly close
        # Map current price to weekly context - use aligned weekly pivot for bias
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # Volume filter: current 1d volume > 1.3x 20-period 1d volume SMA
        # Need to get 1d volume for current day - use aligned array to check if we have volume data
        vol_confirm = True  # default to true if we can't verify, will be overridden below
        
        # Simple volume confirmation: use the aligned volume SMA directly
        # We'll check if current 1d volume exceeds threshold by using the aligned arrays
        # Since we don't have intraday 1d volume, we use the fact that volume tends to persist
        # and check if the volume environment is elevated
        vol_threshold = vol_sma_20_1d_aligned[i] * 1.3
        
        # Approximate current 1d volume using the fact that 6h bars roll into 1d
        # We use the current 6h volume as proxy, scaled appropriately
        # But simpler: just use the volume condition as a regime filter based on recent average
        # Actually, let's use the 1d volume directly from the aligned daily volume series
        # We need to get the 1d volume value - we can approximate by using the aligned volume
        # For now, we'll use a simpler approach: check if volume environment is active
        
        # Use the aligned 1d volume series (we need to extract it)
        # Get 1d OHLCV data
        df_1d_for_vol = get_htf_data(prices, '1d')  # Already fetched above, but we need volume series
        if 'volume' in df_1d_for_vol.columns:
            vol_1d_series = df_1d_for_vol['volume'].values
            vol_1d_aligned = align_htf_to_ltf(prices, df_1d_for_vol, vol_1d_series)
            if not np.isnan(vol_1d_aligned[i]):
                vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # === LONG CONDITIONS ===
        if (close[i] > donchian_upper_aligned[i]) and bullish_bias and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        elif (close[i] < donchian_lower_aligned[i]) and bearish_bias and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_1wPivotBias_1dVol_Filter_v1"
timeframe = "6h"
leverage = 1.0