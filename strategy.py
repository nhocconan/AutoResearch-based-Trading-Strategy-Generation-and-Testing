#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout + 4h volume spike + 1d ADX trend filter
# - Long: price > Camarilla H3 AND 4h volume > 2x 20-period average AND 1d ADX > 20
# - Short: price < Camarilla L3 AND 4h volume > 2x 20-period average AND 1d ADX > 20
# - Uses 1h timeframe for entry timing precision, 4h/1h for confirmation, 1d for trend
# - Camarilla levels calculated from previous day's OHLC (no look-ahead)
# - Discrete position sizing (0.20) to minimize fee churn
# - ATR-based stoploss (1.5x ATR(14)) and time-based exit (48h max hold)
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Target: 15-35 trades/year to avoid fee drag while capturing strong intraday moves

name = "1h_4h_1d_camarilla_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1)) if 'close_4h' in locals() else np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 4h volume confirmation
    volume_4h = df_4h['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume_4h > (2.0 * avg_volume_20)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h.astype(float))
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr1_1h = high_1h - low_1h
    tr2_1h = np.abs(high_1h - np.roll(close_1h, 1))
    tr3_1h = np.abs(low_1h - np.roll(close_1h, 1))
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    tr_1h[0] = tr1_1h[0]
    atr_14 = pd.Series(tr_1h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_bar = 0
    max_hold_bars = 48  # 48 hours max hold
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_spike_4h_aligned[i]) or 
            np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: stoploss hit OR max hold time reached
            if (close_1h[i] < entry_price - 1.5 * atr_14[i]) or (i - entry_bar >= max_hold_bars):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: stoploss hit OR max hold time reached
            if (close_1h[i] > entry_price + 1.5 * atr_14[i]) or (i - entry_bar >= max_hold_bars):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Calculate Camarilla levels from previous day's OHLC
            # Need to get previous day's data - use 1d data shifted by 1
            if i >= 24:  # Need at least 24 hours of 1h data for previous day
                # Get index of 1d bar that completed before current 1h bar
                # Since we're using 1h timeframe, we need to look back 24 bars for previous day
                prev_day_idx = i - 24
                if prev_day_idx >= 0 and prev_day_idx < len(prices):
                    # Use OHLC from 24 hours ago (previous day's close)
                    prev_high = high_1h[prev_day_idx]
                    prev_low = low_1h[prev_day_idx]
                    prev_close = close_1h[prev_day_idx]
                    
                    # Calculate Camarilla levels
                    range_val = prev_high - prev_low
                    if range_val > 0:
                        camarilla_h3 = prev_close + (range_val * 1.1 / 4)
                        camarilla_l3 = prev_close - (range_val * 1.1 / 4)
                        
                        # Look for Camarilla breakout with volume and trend filters
                        if vol_spike_4h_aligned[i] > 0.5 and adx_aligned[i] > 20:
                            # Long: price > Camarilla H3
                            if close_1h[i] > camarilla_h3:
                                position = 1
                                entry_price = close_1h[i]
                                entry_bar = i
                                signals[i] = 0.20
                            # Short: price < Camarilla L3
                            elif close_1h[i] < camarilla_l3:
                                position = -1
                                entry_price = close_1h[i]
                                entry_bar = i
                                signals[i] = -0.20
    
    return signals