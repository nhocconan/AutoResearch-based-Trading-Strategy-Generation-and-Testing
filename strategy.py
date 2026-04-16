#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w volume confirmation and choppiness regime filter.
# Long when KAMA is rising AND price > KAMA AND 1w volume > 1.5x 20-period average AND 1d chop < 61.8 (trending market).
# Short when KAMA is falling AND price < KAMA AND 1w volume > 1.5x 20-period average AND 1d chop < 61.8.
# Exit when price crosses KAMA or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture sustained trends with volume confirmation while avoiding choppy markets.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: KAMA (adaptive trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Pad the beginning with zeros
    er = np.concatenate([np.full(10, np.nan), er])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after first ER calculation
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: rising if current > previous, falling if current < previous
    kama_rising = np.concatenate([[False], np.diff(kama) > 0])
    kama_falling = np.concatenate([[False], np.diff(kama) < 0])
    
    # === 1d Indicators: Choppiness Index (chop < 61.8 = trending) ===
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((hh - ll) > 0, chop, 50)  # Default to middle range when no movement
    
    # === 1w Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
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
    tr1_1d = pd.Series(high_1d).diff()
    tr2_1d = pd.Series(low_1d).diff().abs()
    tr3_1d = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = atr_1d_raw  # Already aligned as primary timeframe
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        kama_val = kama[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        atr_val = atr_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below KAMA
            if price < kama_val:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above KAMA
            if price > kama_val:
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
            # LONG: KAMA rising AND price > KAMA AND volume spike AND trending market (chop < 61.8)
            if kama_rising[i] and price > kama_val and vol_spike and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: KAMA falling AND price < KAMA AND volume spike AND trending market (chop < 61.8)
            elif kama_falling[i] and price < kama_val and vol_spike and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_KAMA_1wVolumeSpike_ChopFilter_V1"
timeframe = "1d"
leverage = 1.0