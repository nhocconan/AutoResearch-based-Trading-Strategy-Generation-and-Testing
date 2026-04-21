#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dChopRegime_VolumeConfirm
Hypothesis: 4h Camarilla R1/S1 breakouts with 1d choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend) and volume confirmation (>1.5x 20-bar MA). 
In ranging markets (CHOP > 61.8), fade breakouts at R1/S1 with mean reversion. In trending markets (CHOP < 38.2), trade breakouts in direction of trend.
Discrete sizing (0.25) and ATR-based stoploss (1.5x) reduce churn. Target: 75-200 total trades over 4 years by using 4h primary timeframe and tight entry conditions requiring confluence of breakout, regime, and volume.
Works in bull (trend-following breakouts) and bear (mean reversion in ranges during choppy regimes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Choppiness Index (CHOP) for regime detection ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over 14 periods
    sum_tr_14 = tr_1d.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_raw = np.where(range_14 > 0, sum_tr_14 / range_14, 1.0)
    chop_1d = 100 * np.log10(chop_raw) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Regime classification
    chop_range = chop_1d_aligned > 61.8  # ranging/mean revert
    chop_trend = chop_1d_aligned < 38.2   # trending
    
    # === 4h ATR (10-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    
    # === 4h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # === 4h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        is_chop_range = chop_range[i]
        is_chop_trend = chop_trend[i]
        vol_conf = volume_confirmed[i]
        r1_val = r1[i]
        s1_val = s1[i]
        
        if position == 0:
            if is_chop_range:
                # Ranging market: mean reversion at extremes
                long_condition = (price < s1_val) and vol_conf  # buy at support
                short_condition = (price > r1_val) and vol_conf  # sell at resistance
            else:  # trending or neutral chop
                # Trending market: breakout in direction of trend (use price action)
                # Simple trend: higher highs/lows via price vs prior pivot
                is_bullish = price > pivot[i]  # bullish bias
                is_bearish = price < pivot[i]  # bearish bias
                long_condition = (price > r1_val) and vol_conf and is_bullish
                short_condition = (price < s1_val) and vol_conf and is_bearish
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (1.5x ATR)
            if position == 1:
                if price < entry_price - 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price fails to hold above S1 (for longs) or breaks R1 (for shorts)
                elif price < s1_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price fails to hold below R1 (for shorts) or breaks S1 (for longs)
                elif price > r1_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dChopRegime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0