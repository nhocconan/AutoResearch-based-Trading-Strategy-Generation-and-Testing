#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike (2.0x) + chop regime filter (CHOP > 61.8)
# Long when price breaks above Donchian upper band + 1d EMA50 uptrend + volume confirmation + choppy market (mean reversion favor)
# Short when price breaks below Donchian lower band + 1d EMA50 downtrend + volume confirmation + choppy market
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Chop regime filter avoids trending markets where breakouts fail; targets ranging/mean-reverting conditions.
# Volume threshold (2.0x) and chop filter target ~30-60 trades/year on 4h to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Chopiness Index (14) on 1d for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log(n))) / log10(n)
    # Simplified: CHOP = 100 * log10( sum(tr) / (max(high)-min(low)) ) / log10(14)
    # where tr = true range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, sum_tr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Donchian(20) channels ===
    # Upper band = max(high, 20)
    # Lower band = min(low, 20)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 14) + 5  # EMA50 + Donchian(20) + Chop(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Chop regime filter: CHOP > 61.8 indicates ranging/market (favorable for mean reversion breakouts)
        chop_filter = chop_aligned[i] > 61.8
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper band (close > upper band)
        # 2. 1d EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        # 4. Chop regime filter (ranging market)
        if (close[i] > highest_high_20[i]) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower band (close < lower band)
        # 2. 1d EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        # 4. Chop regime filter (ranging market)
        elif (close[i] < lowest_low_20[i]) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1dEMA50_Volume_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0