#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
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
    if len(df_1d) < 2:
        return signals
    
    # Calculate daily Camarilla pivot levels (based on previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # L2 = close - 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    
    range_1d = high_1d - low_1d
    
    # Calculate levels using previous day's data to avoid look-ahead
    camarilla_h4 = close_1d + 1.1 * range_1d / 2
    camarilla_l4 = close_1d - 1.1 * range_1d / 2
    camarilla_h3 = close_1d + 1.1 * range_1d / 4
    camarilla_l3 = close_1d - 1.1 * range_1d / 4
    camarilla_h2 = close_1d + 1.1 * range_1d / 6
    camarilla_l2 = close_1d - 1.1 * range_1d / 6
    camarilla_h1 = close_1d + 1.1 * range_1d / 12
    camarilla_l1 = close_1d - 1.1 * range_1d / 12
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h2 = np.roll(camarilla_h2, 1)
    camarilla_l2 = np.roll(camarilla_l2, 1)
    camarilla_h1 = np.roll(camarilla_h1, 1)
    camarilla_l1 = np.roll(camarilla_l1, 1)
    
    # Set first value to NaN (no previous day)
    camarilla_h4[0] = np.nan
    camarilla_l4[0] = np.nan
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    camarilla_h2[0] = np.nan
    camarilla_l2[0] = np.nan
    camarilla_h1[0] = np.nan
    camarilla_l1[0] = np.nan
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: ADX > 20 for trending market (avoid chop)
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
    
    trending_market = adx > 20
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h2 = camarilla_h2_aligned[i]
        l2 = camarilla_l2_aligned[i]
        h1 = camarilla_h1_aligned[i]
        l1 = camarilla_l1_aligned[i]
        adx_val = adx[i]
        trending = trending_market[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma_20[i]
        
        # Entry signals - only in trending markets with volume confirmation
        long_signal = False
        short_signal = False
        
        # Long: price touches or goes below L3 with reversal intention (mean reversion in trend)
        if price_low <= l3 and volume_confirmed and trending:
            long_signal = True
        
        # Short: price touches or goes above H3 with reversal intention (mean reversion in trend)
        if price_high >= h3 and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions
        # Stop loss: 2x ATR from entry
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr[i])
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr[i])
        
        # Take profit: reach opposite H1/L1 level
        tp_long = position == 1 and price_high >= h1
        tp_short = position == -1 and price_low <= l1
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (stop_long or tp_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (stop_short or tp_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla pivot reversal strategy with volume confirmation and trend filter.
# Enters long when price touches or goes below L3 level with volume confirmation in trending markets (ADX > 20).
# Enters short when price touches or goes above H3 level with volume confirmation and ADX > 20.
# Uses daily timeframe for Camarilla pivot levels to capture institutional reversal points.
# Volume confirmation ensures participation, ADX filter avoids false signals in weak trends.
# Exits via ATR stop loss (2.0x) or take profit at H1/L1 levels.
# Designed for 4h timeframe with selective entries to target 75-200 total trades over 4 years.
# Works in both bull and bear markets by trading reversals at key institutional levels.