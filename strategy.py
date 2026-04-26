#!/usr/bin/env python3
"""
1d_KAMA_With_Trend_And_Regime_Filter
Hypothesis: On daily timeframe, KAMA (adaptive trend) identifies market direction, combined with 1-week EMA trend filter and choppiness regime to avoid false signals in ranging markets. Uses volume confirmation to ensure breakout validity. Designed for low trade frequency (target: 30-100 trades over 4 years) to minimize fee drag, works in both bull and bear via adaptive trend and regime filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA on daily prices (ER=10, fast=2, slow=30)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))  # |close_t - close_t-10|
    volatility = np.sum(np.abs(np.subtract(close[1:], close[:-1])), axis=0) if False else None  # placeholder for correct calc
    # Proper volatility calculation: sum of absolute daily changes over 10 periods
    volatility = pd.Series(close).diff().abs().rolling(window=10, min_periods=10).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Pad ER array to match close length (first 10 values undefined)
    er_padded = np.full_like(close, 0.0)
    er_padded[10:] = er
    
    # Smoothing constants: sc = [ER*(2/(fast+1)-2/(slow+1)) + 2/(slow+1)]^2
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = er_padded * (fast_sc - slow_sc) + slow_sc
    sc = sc * sc  # square
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]  # seed
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume confirmation: volume > 1.5x 20-day median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Choppiness regime filter: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hh_ll = highest_high - lowest_low
    # Avoid division by zero and log of non-positive
    chop = np.full_like(close, 50.0)
    mask = (hh_ll > 1e-8) & ~np.isnan(hh_ll) & ~np.isnan(atr) & (atr > 0)
    chop[mask] = 100 * np.log10(atr[mask] / (np.log10(hh_ll[mask]) * 14))
    # Regime: chop between 38.2 and 61.8 = undefined (neutral), <38.2 = trending, >61.8 = ranging
    chop_regime = (chop >= 38.2) & (chop <= 61.8)  # only trade in neutral to mildly trending/choppy to avoid strong ranging
    
    # Fixed position size for low trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 34 for weekly EMA, 30 for KAMA slow, 20 for volume median, 14 for chop/ATR
    start_idx = max(34, 30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(kama[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        ema_34_w_val = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        in_chop_regime = chop_regime[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > KAMA AND price > weekly EMA34 AND volume spike AND in chop regime (not strong ranging)
            long_entry = (close_val > kama_val) and (close_val > ema_34_w_val) and vol_spike and in_chop_regime
            # Short: price < KAMA AND price < weekly EMA34 AND volume spike AND in chop regime
            short_entry = (close_val < kama_val) and (close_val < ema_34_w_val) and vol_spike and in_chop_regime
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price crosses below KAMA OR weekly EMA34 (trend change)
            if (close_val < kama_val) or (close_val < ema_34_w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price crosses above KAMA OR weekly EMA34
            if (close_val > kama_val) or (close_val > ema_34_w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_With_Trend_And_Regime_Filter"
timeframe = "1d"
leverage = 1.0