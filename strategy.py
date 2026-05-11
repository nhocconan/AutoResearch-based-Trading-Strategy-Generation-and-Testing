# ==============================
# EXPERIMENT #151498 - DAILY STRATEGY
# ==============================
# Hypothesis: 1D Donchian(20) breakout with weekly trend filter (1W EMA50 slope) and volume confirmation (1.5x 20D average)
# Uses daily Donchian breakouts for entry, weekly trend for direction filter, and volume surge for confirmation.
# Designed for low trade frequency (7-25/year) to minimize fee drag while capturing strong directional moves.
# Works in both bull and bear markets via trend filter (only trades in direction of weekly trend).
# ==============================

name = "1D_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Donchian(20) channels ---
    # Upper band: 20-day high
    # Lower band: 20-day low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to daily (already daily, but using for consistency)
    # Since we're on daily timeframe, no alignment needed for daily data
    # But we'll keep the structure for clarity
    
    # --- Weekly trend: EMA50 slope ---
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope_50_1w = np.diff(ema_50_1w, prepend=ema_50_1w[0])
    ema_slope_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_50_1w)
    
    # --- ATR for volatility and trailing stop ---
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- Volume confirmation (1.5x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup: ensure we have enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_slope_50_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from weekly EMA50 slope
        bullish_trend = ema_slope_50_1w_aligned[i] > 0
        bearish_trend = ema_slope_50_1w_aligned[i] < 0
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper in bullish trend with volume surge
            if close[i] > donchian_upper[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Donchian lower in bearish trend with volume surge
            elif close[i] < donchian_lower[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 3.0*ATR from highest high
                if close[i] < highest_high_since_entry - 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 3.0*ATR from lowest low
                if close[i] > lowest_low_since_entry + 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals