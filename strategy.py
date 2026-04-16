#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with 1w volume confirmation and 1d chop regime filter.
# Long when KAMA direction is up (price > KAMA) AND 1w volume > 1.5x 20-period 1w average AND 1d chop < 61.8 (trending market).
# Short when KAMA direction is down (price < KAMA) AND 1w volume > 1.5x 20-period 1w average AND 1d chop < 61.8.
# Exit when KAMA direction reverses or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture trends with volume confirmation in non-choppy markets.
# Works in both bull and bear markets by requiring trend (chop<61.8) and volume confirmation, avoiding ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: KAMA (trend direction) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(change)
    er[10:] = change[10:] / volatility[10:]
    er = np.concatenate([np.full(10, np.nan), er])
    # Smoothing Constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = close > kama  # True for up, False for down
    
    # === 1w Indicators: Volume Spike ===
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    # === 1d Indicators: Choppiness Index (regime filter) ===
    # True Range
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((hh_14 - ll_14) == 0, 100, chop)
    chop = np.where(np.isnan(chop), 100, chop)
    trending = chop < 61.8  # Trending market
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 1d ATR for stoploss
    tr1_1d = pd.Series(high).diff()
    tr2_1d = pd.Series(low).diff().abs()
    tr3_1d = pd.Series(close).shift(1).diff().abs()
    tr_1d_raw = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d_raw = pd.Series(tr_1d_raw).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(kama_dir[i]) or np.isnan(volume_spike[i]) or np.isnan(trending[i]) or
            np.isnan(atr_1d_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_trending = trending[i]
        atr_val = atr_1d_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if KAMA direction reverses (price < KAMA)
            if price <= kama[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if KAMA direction reverses (price > KAMA)
            if price >= kama[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: KAMA up (price > KAMA) AND volume spike AND trending market
            if kama_dir[i] and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: KAMA down (price < KAMA) AND volume spike AND trending market
            elif not kama_dir[i] and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_KAMA_1wVolumeSpike_1dChop_V1"
timeframe = "1d"
leverage = 1.0