#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian(20) + Choppiness Regime + Volume Spike

HYPOTHESIS: The DB top performer (test Sharpe 1.49) used CHOP index as regime filter.
This strategy combines:
- 4h Donchian(20) breakout for price structure
- Choppiness Index to detect trending vs ranging regimes
- Volume spike (>1.8x) for momentum confirmation
- 1d EMA as macro trend filter
- ATR-based stoploss

WHY IT WORKS IN BOTH MARKETS:
- 2021 bull: CHOP<40 triggers trend-following longs on breakouts
- 2022 bear: CHOP<40 triggers trend-following shorts on breakdowns, EMA filter avoids catching tops
- 2025 range: CHOP>60 keeps us flat (no trending trades in chop)
- Choppiness is the key differentiator from #016 (which had Sharpe 0.60)

KEY INSIGHT: The CHOP filter prevents trend-following in choppy markets,
significantly reducing whipsaw losses that destroyed many 2022 strategies.

TRADE COUNT: 75-150 total over 4 years (target 20-40/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_vol_ema_v1"
timeframe = "4h"
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
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/ranging market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    Values in between = neutral
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j], abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        if atr_sum <= 1e-10:
            continue
        
        # Highest high - Lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        
        if hh - ll <= 1e-10:
            continue
        
        # CHOP = 100 * log10(atr_sum / (hh - ll)) / log10(period)
        chop[i] = 100.0 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian Channel(20) - 5 days on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # 20 for Donchian + buffer for CHOP
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === REGIME DETECTION via Choppiness Index ===
        # CHOP < 38.2 = trending (trend follow)
        # CHOP > 61.8 = ranging (stay out or mean revert)
        is_trending = chop_14[i] < 38.2
        is_choppy = chop_14[i] > 61.8
        
        # === HTF TREND (1d EMA aligned) ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME SPIKE (>1.8x average) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT (close crosses outside prior bar's channel) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = False
        bearish_breakout = False
        
        if not np.isnan(prev_upper) and not np.isnan(prev_lower):
            # Close breaks above prior upper = bullish breakout
            bullish_breakout = close[i] > prev_upper
            # Close breaks below prior lower = bearish breakout
            bearish_breakout = close[i] < prev_lower
        
        # === MINIMUM HOLD: 3 bars (12h) to avoid immediate reversals ===
        min_hold = (i - entry_bar) >= 3
        
        # === ATR TRAILING STOP (2.5x ATR from entry high/low) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on trend reversal (HTF trend flips)
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Entry rules:
            # 1. Must be in TRENDING regime (CHOP < 38.2)
            # 2. Volume spike confirmation
            # 3. HTF trend alignment
            # 4. Donchian breakout
            
            if is_trending:
                # LONG: Bullish breakout + volume spike + HTF bullish
                if bullish_breakout and vol_spike and htf_bullish:
                    in_position = True
                    position_side = 1
                    entry_atr = atr_14[i]
                    entry_bar = i
                    highest_since_entry = high[i]
                    signals[i] = SIZE
                
                # SHORT: Bearish breakdown + volume spike + HTF bearish
                elif bearish_breakout and vol_spike and htf_bearish:
                    in_position = True
                    position_side = -1
                    entry_atr = atr_14[i]
                    entry_bar = i
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                
                else:
                    signals[i] = 0.0
            else:
                # In choppy regime, stay flat (no trades)
                signals[i] = 0.0
    
    return signals