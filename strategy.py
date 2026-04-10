#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume filter and ATR stoploss
# - Uses 1d Camarilla levels (H3, L3) as key support/resistance from daily structure
# - Long when price breaks above H3 with volume confirmation (>1.5x 20-period avg volume)
# - Short when price breaks below L3 with volume confirmation
# - 1d ADX > 20 ensures we only trade when daily timeframe has sufficient trend strength
# - ATR-based trailing stop: exit when price moves against position by 2.0*ATR
# - Designed for 4h timeframe: targets 20-40 trades/year to avoid fee drag
# - Camarilla pivots work well in both trending and ranging markets due to their mathematical derivation
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for each day (using previous day's data)
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # We shift by 1 to use previous day's levels for today's trading
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_open = np.roll(open_1d, 1)
    
    # First day has no previous data
    prev_high[0] = prev_low[0] = prev_close[0] = prev_open[0] = 0
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_h3 = prev_close + 1.0 * range_1d  # H3 level
    camarilla_l3 = prev_close - 1.0 * range_1d  # L3 level
    camarilla_h4 = prev_close + 1.5 * range_1d  # H4 level (stronger resistance)
    camarilla_l4 = prev_close - 1.5 * range_1d  # L4 level (stronger support)
    
    # Pre-compute 1d ADX(14) for trend filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * avg_volume_20)
    
    # Pre-compute 4h ATR(14) for trailing stop
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0  # for trailing stop
    lowest_low = 0.0    # for trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            if close_4h[i] > highest_high:
                highest_high = close_4h[i]
            # Exit: trailing stop hit OR price breaks below L3 (failed breakout)
            if close_4h[i] < highest_high - 2.0 * atr_14[i] or close_4h[i] < l3_aligned[i]:
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            if close_4h[i] < lowest_low:
                lowest_low = close_4h[i]
            # Exit: trailing stop hit OR price breaks above H3 (failed breakout)
            if close_4h[i] > lowest_low + 2.0 * atr_14[i] or close_4h[i] > h3_aligned[i]:
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 20:
                # Breakout long: price closes above H3
                if close_4h[i] > h3_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    highest_high = close_4h[i]
                    signals[i] = 0.25
                # Breakout short: price closes below L3
                elif close_4h[i] < l3_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    lowest_low = close_4h[i]
                    signals[i] = -0.25
    
    return signals