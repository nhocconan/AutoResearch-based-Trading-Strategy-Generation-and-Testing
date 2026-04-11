#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return signals
    
    # Calculate daily Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels for each day
    # Using previous day's data to avoid look-ahead
    range_1d = high_1d - low_1d
    # Shift by 1 to use previous day's data
    range_1d_prev = np.roll(range_1d, 1)
    range_1d_prev[0] = 0  # First day has no previous
    
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]
    
    # Camarilla levels: H4, L4 (main levels)
    H4 = close_1d_prev + 1.1 * range_1d_prev / 2
    L4 = close_1d_prev - 1.1 * range_1d_prev / 2
    
    # Align daily Camarilla to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: volume > 1.5x 20-period average (stricter filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: ADX > 25 for trending market
    # Calculate ADX components
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Smooth +DM and -DM
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    trending_market = adx > 25
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        H4 = H4_aligned[i]
        L4 = L4_aligned[i]
        adx_val = adx[i]
        trending = trending_market[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Entry signals - only in trending markets
        long_signal = False
        short_signal = False
        
        # Long: price breaks above H4 with volume and trend
        if price_high > H4 and volume_confirmed and trending:
            long_signal = True
        
        # Short: price breaks below L4 with volume and trend
        if price_low < L4 and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions
        # Calculate midpoint for exit (using previous day's close)
        midpoint_prev = close_1d_prev
        midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_prev)[i]
        
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 1.5 * atr[i])
        stop_short = position == -1 and price_high > (entry_price + 1.5 * atr[i])
        
        # Exit to midpoint
        exit_long = position == 1 and price_close < midpoint_aligned
        exit_short = position == -1 and price_close > midpoint_aligned
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla breakout strategy with stricter volume confirmation and ADX trend filter.
# Enters long when price breaks above daily Camarilla H4 level with volume confirmation (>1.5x avg volume) in trending markets (ADX > 25).
# Enters short when price breaks below daily Camarilla L4 level with volume confirmation and ADX > 25.
# Uses previous day's data to calculate Camarilla levels to avoid look-ahead.
# Volume confirmation ensures institutional participation, ADX filter avoids whipsaws in sideways markets.
# Exits when price returns to previous day's close or ATR stop loss (1.5x) is hit.
# Designed for 4h timeframe with tighter entry conditions to target 75-200 total trades over 4 years.
# Works in both bull and bear markets by trading breakouts in either direction with trend filter.