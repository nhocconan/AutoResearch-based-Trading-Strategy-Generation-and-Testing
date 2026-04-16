#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d regime filter (ADX + Chop)
# Long when 6h Bear Power turns up from negative AND 1d ADX > 25 AND 1d Chop > 61.8 (trending but not too choppy)
# Short when 6h Bull Power turns down from positive AND 1d ADX > 25 AND 1d Chop > 61.8
# Exit when Elder Power reverses or ATR-based stop (2*ATR from entry)
# Uses discrete position size 0.25. Designed to catch momentum shifts in established trends.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray (Bull Power, Bear Power) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # EMA13 for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_6h = high_6h - ema13_6h
    # Bear Power = Low - EMA13
    bear_power_6h = low_6h - ema13_6h
    
    # Align 6h indicators to primary timeframe (already 6h)
    bull_power_6h_aligned = bull_power_6h
    bear_power_6h_aligned = bear_power_6h
    
    # === 1d Indicators: ADX (trend strength) and Chop (choppiness) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Chop Index: measures choppiness vs trending
    # High Chop (>61.8) = ranging, Low Chop (<38.2) = trending
    # We want Chop > 61.8 to ensure we're not in extreme chop (but still trending via ADX)
    atr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_sum_1d / (max_high_1d - min_low_1d)) / np.log10(14)
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 6h ATR for stoploss ===
    tr1_6h = pd.Series(high_6h).diff()
    tr2_6h = pd.Series(low_6h).diff().abs()
    tr3_6h = pd.Series(close_6h).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Already aligned as primary timeframe
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power_6h_aligned[i]) or np.isnan(bear_power_6h_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(atr_6h[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bull_power = bull_power_6h_aligned[i]
        bear_power = bear_power_6h_aligned[i]
        adx_val = adx_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        atr_val = atr_6h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bear Power turns up from negative (momentum fading)
            if i > warmup and bear_power_6h_aligned[i-1] < 0 and bear_power >= 0:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bull Power turns down from positive (momentum fading)
            if i > warmup and bull_power_6h_aligned[i-1] > 0 and bull_power <= 0:
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
            # Regime filter: trending but not too choppy (ADX>25 and Chop>61.8)
            is_trending_regime = adx_val > 25 and chop_val > 61.8
            
            # LONG: Bear Power turns up from negative AND trending regime
            if i > warmup and bear_power_6h_aligned[i-1] < 0 and bear_power >= 0 and is_trending_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bull Power turns down from positive AND trending regime
            elif i > warmup and bull_power_6h_aligned[i-1] > 0 and bull_power <= 0 and is_trending_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dADXChop_V1"
timeframe = "6h"
leverage = 1.0