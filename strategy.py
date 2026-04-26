#!/usr/bin/env python3
"""
1d_KAMA_Regime_DonchianBreakout_1wTrend
Hypothesis: KAMA (10,2,30) identifies adaptive trend direction on 1d. Donchian(20) breakout provides entry timing with 1w EMA50 trend filter. Volume confirmation (>1.5x mean volume) and choppiness regime (CHOP > 61.8 = range) filter false signals. Discrete sizing 0.25 limits trades to ~10-20/year. Works in bull/bear via 1w trend filter and volatility-adjusted exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for KAMA and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d (ER=10, FAST=2, SLOW=30)
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[29] = close_1d[29]  # seed
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate Donchian(20) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for volatility filtering and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volume confirmation: >1.5x 20-period mean volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)
    
    # Choppiness regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
    # We'll use CHOP as regime filter: only trade when CHOP > 50 (range-bound) for mean reversion
    # Calculate CHOP on 1d
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(atr_sum / np.log(10) / (highest_high - lowest_low)) / np.log10(14), 
                    50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_ok = chop_aligned > 50  # range regime
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d KAMA, 20 for Donchian, 50 for 1w EMA, 14 for ATR/CHOP
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_1d_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_ok = volume_ok[i]
        chop_ok_val = chop_ok[i]
        atr_val = atr[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above Donchian high in uptrend OR breaks below Donchian low in downtrend (mean reversion in range)
            # Short: price breaks below Donchian low in downtrend OR breaks above Donchian high in uptrend (mean reversion in range)
            long_breakout = close_val > donchian_high_val and close_val > ema_50_val
            short_breakout = close_val < donchian_low_val and close_val < ema_50_val
            long_mean_revert = close_val < donchian_low_val and chop_ok_val and close_val < kama_val
            short_mean_revert = close_val > donchian_high_val and chop_ok_val and close_val > kama_val
            
            long_entry = (long_breakout or long_mean_revert) and vol_ok
            short_entry = (short_breakout or short_mean_revert) and vol_ok
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal, mean reversion, or ATR stop
            # Exit if: trend turns bearish OR price reverts to KAMA OR 2*ATR stop loss
            if (close_val < ema_50_val or 
                abs(close_val - kama_val) < 0.5 * atr_val or 
                close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, mean reversion, or ATR stop
            # Exit if: trend turns bullish OR price reverts to KAMA OR 2*ATR stop loss
            if (close_val > ema_50_val or 
                abs(close_val - kama_val) < 0.5 * atr_val or 
                close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Regime_DonchianBreakout_1wTrend"
timeframe = "1d"
leverage = 1.0