#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation + 1w Supertrend filter
# - Primary signal: Long when 6h price closes below VWAP by 1.5 ATR AND 1w Supertrend is bullish
# - Short when 6h price closes above VWAP by 1.5 ATR AND 1w Supertrend is bearish
# - VWAP reset daily, ATR calculated on 6h timeframe
# - Supertrend uses ATR(10) multiplier 3.0 on weekly timeframe for robust trend filter
# - Works in bull/bear: VWAP mean reversion captures short-term extremes, Supertrend ensures
#   alignment with higher timeframe trend to avoid counter-trend whipsaws
# - Target: 12-37 trades/year (50-150 total over 4 years) via tight entry conditions

name = "6h_1w_vwap_supertrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1w Supertrend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR calculation for Supertrend
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - pd.Series(close_1w).shift(1)))
    tr3 = pd.Series(np.abs(low_1w - pd.Series(close_1w).shift(1)))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_1w = (high_1w + low_1w) / 2
    upper_band_1w = hl2_1w + (3.0 * atr_1w)
    lower_band_1w = hl2_1w - (3.0 * atr_1w)
    
    supertrend_1w = np.zeros(len(close_1w))
    direction_1w = np.ones(len(close_1w))  # 1 for uptrend, -1 for downtrend
    
    supertrend_1w[0] = upper_band_1w[0]
    direction_1w[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend_1w[i-1]:
            supertrend_1w[i] = upper_band_1w[i]
            direction_1w[i] = 1
        else:
            supertrend_1w[i] = lower_band_1w[i]
            direction_1w[i] = -1
    
    # Align 1w Supertrend direction to 6h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h ATR for VWAP bands
    tr1_6h = pd.Series(high - low)
    tr2_6h = pd.Series(np.abs(high - pd.Series(close).shift(1)))
    tr3_6h = pd.Series(np.abs(low - pd.Series(close).shift(1)))
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h = tr_6h.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily VWAP calculation (reset each day)
    # Extract date from open_time for daily grouping
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    vwap = np.full(n, np.nan)
    cumulative_tpv = 0.0
    cumulative_volume = 0.0
    current_date = dates[0] if len(dates) > 0 else None
    
    for i in range(n):
        if dates[i] != current_date:
            # Reset for new day
            cumulative_tpv = 0.0
            cumulative_volume = 0.0
            current_date = dates[i]
        
        typical_price = (high[i] + low[i] + close[i]) / 3.0
        cumulative_tpv += typical_price * volume[i]
        cumulative_volume += volume[i]
        
        if cumulative_volume > 0:
            vwap[i] = cumulative_tpv / cumulative_volume
    
    # VWAP deviation in ATR units
    vwap_deviation = (close - vwap) / atr_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend_dir_aligned[i]) or
            np.isnan(vwap_deviation[i]) or
            np.isnan(atr_6h[i]) or
            atr_6h[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to VWAP OR Supertrend turns bearish
            if vwap_deviation[i] >= -0.5 or supertrend_dir_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to VWAP OR Supertrend turns bullish
            if vwap_deviation[i] <= 0.5 or supertrend_dir_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for VWAP extremes with Supertrend filter
            # Long: price significantly below VWAP AND Supertrend bullish
            if vwap_deviation[i] <= -1.5 and supertrend_dir_aligned[i] == 1:
                position = 1
                signals[i] = 0.25
            # Short: price significantly above VWAP AND Supertrend bearish
            elif vwap_deviation[i] >= 1.5 and supertrend_dir_aligned[i] == -1:
                position = -1
                signals[i] = -0.25
    
    return signals