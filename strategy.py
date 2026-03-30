#!/usr/bin/env python3
"""
Experiment #025: 12h Donchian + Weekly Trend + Volume + Choppiness Regime

HYPOTHESIS: 12h timeframe with choppiness regime filter solves the 2022 bear market
problem that killed previous 12h strategies. Choppiness < 50 = trending = safe to enter.
Choppiness > 61.5 = ranging = skip (avoids 2022 whipsaw that destroyed #017).

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull (2021, 2023-24): Breakout above 20-bar high + volume spike + trending regime = ride the move
- Bear (2022): Choppiness stays HIGH in ranging crash → NO entries during whipsaw
- Range (2025): Choppiness also high → still no entries = preserved capital

The key fix from failed #017: adding Choppiness regime filter. #017 had only 43 trades but
negative Sharpe because it entered during the 2022 chop/range. This version SKIPS 2022.

EXPECTED TRADES: ~100-200 total over 4 years (25-50/year per symbol)
- 12h Donchian(20) ≈ 11 breakouts/year per symbol
- Volume spike (1.5x) → 40% pass = ~7/year
- Weekly EMA trend → 50% pass = ~4/year
- Choppiness < 50 regime → 70% pass = ~3/year
- Plus re-entries and early exits → ~6-8/year = 24-32/year per symbol
- 3 symbols × 4 years × 30/year = ~360 total (upper bound)
- With min-hold 6 bars + tighter exit = ~200-250 total (acceptable)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_weekly_vol_chop_v1"
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
    """Choppiness Index - values < 50 = trending, > 61.5 = ranging"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        # Highest high - lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(tr_sum) / np.log10(range_sum))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend
    weekly_ema = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 50  # Enough for Donchian20, ATR14, Choppiness14
    
    for i in range(warmup, n):
        # Check required data
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        desired_signal = 0.0
        
        # === REGIME CHECK: Only trade when trending ===
        trending_regime = chop[i] < 50.0
        
        # === TREND DIRECTION: Weekly EMA ===
        bull_trend = close[i] > weekly_ema_aligned[i]
        bear_trend = close[i] < weekly_ema_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           high[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           low[i] < prev_donchian_low)
        
        # === ENTRY LOGIC ===
        if not in_position:
            # LONG: Trending regime + bullish breakout + volume spike + bull trend
            if trending_regime and bullish_breakout and vol_spike and bull_trend:
                desired_signal = SIZE
            
            # SHORT: Trending regime + bearish breakout + volume spike + bear trend
            elif trending_regime and bearish_breakout and vol_spike and bear_trend:
                desired_signal = -SIZE
        
        # === EXIT/STOPS LOGIC ===
        if in_position:
            if position_side > 0:
                # Trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.5 ATR from highest
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if weekly trend flips
                elif close[i] < weekly_ema_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.5 ATR from lowest
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if weekly trend flips
                elif close[i] > weekly_ema_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 6 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals