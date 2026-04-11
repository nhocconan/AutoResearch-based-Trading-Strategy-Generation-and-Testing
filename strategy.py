#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_v1"
timeframe = "12h"
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
    if len(df_1d) < 2:
        return signals
    
    # Calculate daily Camarilla pivot levels from previous day's data
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    range_ = prev_high - prev_low
    camarilla_h4 = prev_close + range_ * 1.1 / 2
    camarilla_l4 = prev_close - range_ * 1.1 / 2
    camarilla_h3 = prev_close + range_ * 1.1 / 4
    camarilla_l3 = prev_close - range_ * 1.1 / 4
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: ADX > 20 for trending market
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
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    trending_market = adx > 20
    
    for i in range(50, n):
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
        adx_val = adx[i]
        trending = trending_market[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma_20[i]
        
        # Entry signals - trade reversions at Camarilla levels
        long_signal = False
        short_signal = False
        
        # Long: price touches or breaks below L3/L4 with volume confirmation
        if price_low <= l4 and volume_confirmed and trending:
            long_signal = True
        
        # Short: price touches or breaks above H3/H4 with volume confirmation
        if price_high >= h4 and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions - mean reversion to midpoint
        midpoint = (h4 + l4) / 2.0
        exit_long = position == 1 and price_close >= midpoint
        exit_short = position == -1 and price_close <= midpoint
        
        # Stop loss: 1.5x ATR from entry
        stop_long = position == 1 and price_low < (entry_price - 1.5 * atr[i])
        stop_short = position == -1 and price_high > (entry_price + 1.5 * atr[i])
        
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

# Hypothesis: Camarilla pivot reversal strategy on 12h timeframe with daily pivot levels.
# Enters long when price touches or breaks below Camarilla L4 level with volume confirmation in trending markets.
# Enters short when price touches or breaks above Camarilla H4 level with volume confirmation in trending markets.
# Uses volume confirmation (>1.3x 20-period average) to ensure institutional participation.
# Uses ADX > 20 to filter for trending markets and avoid whipsaws in sideways conditions.
# Exits when price returns to the midpoint between H4 and L4 levels.
# Stop loss at 1.5x ATR from entry point.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by trading reversions at key intraday levels derived from prior day's range.