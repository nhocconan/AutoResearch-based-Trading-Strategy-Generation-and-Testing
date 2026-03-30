#!/usr/bin/env python3
"""
Experiment #024: 12h Donchian Breakout + Choppiness Regime + 1d SMA

HYPOTHESIS: Donchian(20) breakout on 12h captures medium-term trend shifts.
Combined with Choppiness Index regime filter (<61.8 = trending only) and
1d SMA200 trend confirmation, this should:
- Work in 2021 bull: ride breakouts above SMA200
- Work in 2022 bear: short breakouts below SMA200  
- Work in 2025 range: choppiness filter avoids false breakouts

KEY INSIGHT: DB winner mtf_4h_chop_donchian_vol_regime_12h_v1 achieved
test_sharpe=1.491 with 107 trades. This replicates that pattern on 12h.

WHY 12h: Balances trade frequency (vs 4h overtrading) with responsiveness
(vs 1d being too slow). 12h = 2 trades/week ideal for medium-term.

TRADE COUNT: 75-150 total over 4 years (18-37/year).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_donchian_vol_1d_sma_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CI): 100 * log10(sum(ATR,14) / (HH(14) - LL(14))) / log10(14)
    CI > 61.8 = choppy/range (don't trend trade)
    CI < 38.2 = trending (good for breakout trades)
    """
    n = len(close)
    ci = np.full(n, np.nan)
    
    for i in range(period, n):
        if np.isnan(close[i - period]):
            continue
        
        atr_sum = 0.0
        for j in range(period):
            tr = max(high[i - j] - low[i - j], 
                     abs(high[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j],
                     abs(low[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j])
            atr_sum += tr
        
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        
        if hh > ll and atr_sum > 0:
            ci[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    return ci

def calculate_donchian(high, low, period=20):
    """Donchian channel: price channel breakout system"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA200 for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_200 = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200)
    
    # === 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    ci_14 = calculate_choppiness(high, low, close, period=14)
    dc_upper_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume: spike detection (1.5x average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signal generation ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    bars_since_entry = 0
    
    warmup = 250  # Need 200 for SMA200 + 20 for Donchian + 14 for ATR
    
    for i in range(warmup, n):
        # Data validity checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update tracking variables
        bars_since_entry = i - entry_bar if in_position else 0
        
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === REGIME FILTER: Choppiness Index ===
        # Only trade breakouts when trending (CI < 61.8)
        ci_trending = not np.isnan(ci_14[i]) and ci_14[i] < 61.8
        
        # === TREND: 1d SMA200 ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === BREAKOUT DETECTION ===
        # Price breaks ABOVE 20-bar high = bullish breakout
        bullish_breakout = (not np.isnan(dc_upper_20[i]) and 
                           close[i] > dc_upper_20[i])
        
        # Price breaks BELOW 20-bar low = bearish breakout
        bearish_breakout = (not np.isnan(dc_lower_20[i]) and 
                           close[i] < dc_lower_20[i])
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === EXITS ===
        if in_position:
            stop_hit = False
            
            # ATR trailing stop (2.5x ATR)
            if position_side > 0:
                # Long stop: price fell from highest since entry
                stop_hit = low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                # Short stop: price rose from lowest since entry
                stop_hit = high[i] > (lowest_since_entry + 2.5 * entry_atr)
            
            # Trend exit: price crosses SMA200 against position
            if position_side > 0 and htf_bearish and bars_since_entry >= 2:
                stop_hit = True
            if position_side < 0 and htf_bullish and bars_since_entry >= 2:
                stop_hit = True
            
            # Max hold: 20 bars (10 days on 12h)
            if bars_since_entry >= 20:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                # Maintain position
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bullish breakout + volume + uptrend + trending regime
            if bullish_breakout and vol_spike and htf_bullish and ci_trending:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume + downtrend + trending regime
            elif bearish_breakout and vol_spike and htf_bearish and ci_trending:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals