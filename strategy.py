#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    if len(df_1d) < 5:
        return signals
    
    # Calculate daily Camarilla pivot levels
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots for previous day (shift by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first value
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    range_val = prev_high - prev_low
    camarilla_h4 = prev_close + range_val * 1.1 / 2
    camarilla_l4 = prev_close - range_val * 1.1 / 2
    camarilla_h3 = prev_close + range_val * 1.1 / 4
    camarilla_l3 = prev_close - range_val * 1.1 / 4
    camarilla_h2 = prev_close + range_val * 1.1 / 6
    camarilla_l2 = prev_close - range_val * 1.1 / 6
    camarilla_h1 = prev_close + range_val * 1.1 / 12
    camarilla_l1 = prev_close - range_val * 1.1 / 12
    
    # Align to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Volume confirmation: volume > 1.5x 12-period average (on 12h timeframe)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # Trend filter: ADX > 25 for trending market (calculated on 12h timeframe)
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
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma_12[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        h2 = h2_aligned[i]
        l2 = l2_aligned[i]
        h1 = h1_aligned[i]
        l1 = l1_aligned[i]
        adx_val = adx[i]
        trending = trending_market[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_12[i]
        
        # Entry signals - only in trending markets
        long_signal = False
        short_signal = False
        
        # Long: price breaks above H3 level with volume and trend
        if price_high > h3 and volume_confirmed and trending:
            long_signal = True
        
        # Short: price breaks below L3 level with volume and trend
        if price_low < l3 and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr[i])
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr[i])
        
        # Exit when price returns to L4/H4 levels (mean reversion within trend)
        exit_long = position == 1 and price_close < l4
        exit_short = position == -1 and price_close > h4
        
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

# Hypothesis: Camarilla breakout strategy with volume confirmation and ADX trend filter on 12h timeframe.
# Enters long when price breaks above Camarilla H3 level with volume confirmation (>1.5x avg volume) in trending markets (ADX > 25).
# Enters short when price breaks below Camarilla L3 level with volume confirmation and ADX > 25.
# Uses daily timeframe for Camarilla levels to capture multi-day swing points.
# Volume confirmation ensures institutional participation, ADX filter avoids whipsaws in sideways markets.
# Exits when price returns to the L4/L4 levels or ATR stop loss (2.0x) is hit.
# Designed for 12h timeframe with tight entry conditions to target 50-150 total trades over 4 years.
# Works in both bull and bear markets by trading breakouts in either direction with trend filter.