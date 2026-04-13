#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter.
    # Uses discrete position size 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.
    # Works in bull markets (breakouts above H3/H4) and bear markets (breakouts below L3/L4) by confirming with 12h volume spike and chop regime.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation and volume (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for volume confirmation (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels from previous day (using prior 4h bar's daily OHLC)
    # We'll use the prior completed 4h bar's price to calculate daily-like pivot for 4h timeframe
    # For simplicity, we use prior 4h bar's high, low, close to calculate Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate prior bar's pseudo-daily OHLC for Camarilla
    # We'll use rolling window of 6 bars (since 6*4h = 24h approx) to get daily-like range
    # But to avoid look-ahead, we use shift(1) on the rolling window
    roll_high = pd.Series(high_4h).rolling(window=6, min_periods=6).max().shift(1).values
    roll_low = pd.Series(low_4h).rolling(window=6, min_periods=6).min().shift(1).values
    roll_close = pd.Series(close_4h).rolling(window=6, min_periods=6).last().shift(1).values
    
    # Camarilla levels calculation
    # H4 = close + 1.5*(high-low)
    # L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low)
    # L3 = close - 1.125*(high-low)
    # We'll use H4/L3 for long entries and L4/H3 for short entries
    high_low = roll_high - roll_low
    h4 = roll_close + 1.5 * high_low
    l4 = roll_close - 1.5 * high_low
    h3 = roll_close + 1.125 * high_low
    l3 = roll_close - 1.125 * high_low
    
    # Calculate 12h volume mean (20-period) with min_periods for confirmation
    volume_12h_series = pd.Series(df_12h['volume'].values)
    vol_ma_20_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    vol_12h_raw = df_12h['volume'].values
    
    # Calculate 4h choppiness index (CHOP) regime filter
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    # We'll use CHOP > 50 as ranging filter for mean reversion at pivot levels
    tr_4h = np.maximum(high_4h - low_4h, 
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), 
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    # Handle first bar
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_4h * 14)) / np.log10(10)
    # For simplicity, we'll use chop > 50 as ranging market
    
    # Align HTF indicators to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_raw)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(vol_12h_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_12h_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Regime filter: chop > 50 indicates ranging market (good for mean reversion at pivots)
        ranging_market = chop_aligned[i] > 50
        
        # Entry conditions: price touches Camarilla levels with volume confirmation and ranging market
        # Long when price touches L3 (support) in ranging market with volume spike
        long_entry = (close[i] <= l3_aligned[i] and volume_confirmation and ranging_market)
        # Short when price touches H3 (resistance) in ranging market with volume spike
        short_entry = (close[i] >= h3_aligned[i] and volume_confirmation and ranging_market)
        
        # Exit conditions: price moves to opposite H3/L3 level or midpoint
        long_exit = close[i] >= h3_aligned[i]  # Exit long at resistance
        short_exit = close[i] <= l3_aligned[i]  # Exit short at support
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_4h_12h_camarilla_pivot_volume_chop_v1"
timeframe = "4h"
leverage = 1.0