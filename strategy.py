#!/usr/bin/env python3
"""
1d_KAMA_Regime_Filter_DonchianExit
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with choppiness index regime filter to avoid whipsaws, and Donchian(20) breakouts
for precise entry timing. Uses volume confirmation (>1.5x 20-day average) to filter low-quality
breakouts. ATR-based stoploss (2.5x) and discrete sizing (0.25). Designed for low trade frequency
(15-25/year) to minimize fee drag while capturing major trends in both bull and bear markets.
KAMA adapts to market noise, making it effective in ranging conditions, while the chop filter
ensures we only trend-follow when markets are truly trending.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop for regime context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly EMA20 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Daily OHLC for indicators ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # === Daily KAMA (ER=10, FAST=2, SLOW=30) for trend direction ===
    close_s = pd.Series(df_1d_close)
    change = close_s.diff(10).abs()
    volatility = close_s.diff().abs().rolling(10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [np.nan] * len(df_1d_close)
    kama[0] = df_1d_close[0]
    for i in range(1, len(df_1d_close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (df_1d_close[i] - kama[i-1])
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === Daily Choppiness Index (14-period) for regime filter ===
    high_low = df_1d_high - df_1d_low
    high_pc = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    low_pc = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(high_low, np.maximum(high_pc, low_pc))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (max_high - min_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Daily Donchian(20) for entry/exit ===
    highest_20 = pd.Series(df_1d_high).rolling(window=20, min_periods=20).max()
    lowest_20 = pd.Series(df_1d_low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_20.values
    donchian_low = lowest_20.values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === Daily volume confirmation (>1.5x 20-day average) ===
    vol_ma = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # === Daily ATR(14) for stoploss ===
    tr1 = pd.Series(df_1d_high - df_1d_low)
    tr2 = pd.Series(np.abs(df_1d_high - np.roll(df_1d_close, 1)))
    tr3 = pd.Series(np.abs(df_1d_low - np.roll(df_1d_close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = df_1d_close[i]
        volume_now = df_1d_volume[i]
        kama_val = kama_aligned[i]
        chop_val = chop_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_avg = vol_ma_aligned[i]
        atr_val = atr_aligned[i]
        weekly_ema = ema_20_1w_aligned[i]
        
        # Regime filter: only trend-follow when chop < 50 (trending market)
        trending_regime = chop_val < 50
        
        # Volume confirmation: >1.5x average volume
        volume_confirm = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Enter long: price breaks above Donchian high + above KAMA + weekly uptrend + volume + regime
            long_condition = (price > donchian_high_val) and (price > kama_val) and \
                           (weekly_ema > np.roll(ema_20_1w_aligned, 1)[i] if i > 0 else True) and \
                           trending_regime and volume_confirm
            
            # Enter short: price breaks below Donchian low + below KAMA + weekly downtrend + volume + regime
            short_condition = (price < donchian_low_val) and (price < kama_val) and \
                            (weekly_ema < np.roll(ema_20_1w_aligned, 1)[i] if i > 0 else True) and \
                            trending_regime and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: stoploss, Donchian low break, or regime change to choppy
            stoploss = price < entry_price - 2.5 * atr_val
            donchian_break = price < donchian_low_val
            regime_change = chop_val >= 55  # hysteresis to avoid whipsaw
            
            if stoploss or donchian_break or regime_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: stoploss, Donchian high break, or regime change to choppy
            stoploss = price > entry_price + 2.5 * atr_val
            donchian_break = price > donchian_high_val
            regime_change = chop_val >= 55  # hysteresis to avoid whipsaw
            
            if stoploss or donchian_break or regime_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Filter_DonchianExit"
timeframe = "1d"
leverage = 1.0