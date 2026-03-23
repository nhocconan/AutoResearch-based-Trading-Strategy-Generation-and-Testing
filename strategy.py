#!/usr/bin/env python3
"""
Experiment #690: 1h Primary + 4h/12h HTF — Regime-Adaptive RSI + HMA Trend

Hypothesis: After 600+ failed strategies, the pattern for 1h is clear:
1. CRSI is too complex for 1h - simpler RSI(14) works better with proper thresholds
2. Previous 1h attempts (#680, #685) failed due to TOO STRICT entries = 0 trades
3. Session filters kill trade frequency on 1h - remove them
4. Need LOOSER entry thresholds (RSI 30/70 not 10/90) to ensure trade generation
5. Use 4h HMA for intermediate trend (not 1d which is too slow for 1h entries)
6. Use 12h HMA for major bias filter only

This strategy uses:
- Choppiness Index (14) for regime: >55 = range (mean revert), <45 = trend (follow)
- RSI(14) for entries: <35 long in range, >65 short in range
- 4h HMA(21) for intermediate trend direction
- 12h HMA(21) for major trend bias (only filter, not entry trigger)
- Volume confirmation: >0.7x 20-bar avg (loose filter)

Why this might work when #680/#685 failed:
- Simpler RSI(14) instead of complex CRSI
- LOOSER thresholds (35/65 vs 10/90) = more trades
- No session filter = trades throughout day
- 4h HMA (not 1d) = better alignment with 1h entries
- Conservative size 0.25 for 1h timeframe

Position sizing: 0.25 discrete (lower for 1h per Rule 10)
Target: 40-80 trades/year on 1h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_rsi_hma_4h12h_v2"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8: Range | CHOP < 38.2: Trend
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h = calculate_hma(close, period=21)
    
    # Volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, lower for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1h[i]) or np.isnan(vol_avg_20[i]):
            continue
        if atr_14[i] == 0 or vol_avg_20[i] == 0:
            continue
        
        # === HTF TREND BIAS ===
        # 4h HMA slope (3 bars)
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3]
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3]
        
        # 12h HMA slope (2 bars) - major trend
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-2]
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-2]
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 50.0  # Range/consolidation (looser threshold)
        is_trend = chop_14[i] < 45.0  # Trending
        
        # === RSI SIGNALS (looser thresholds for trade generation) ===
        rsi_oversold = rsi_14[i] < 40.0  # Was 35, loosened for more trades
        rsi_overbought = rsi_14[i] > 60.0  # Was 65, loosened for more trades
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === VOLUME CONFIRMATION (loose filter) ===
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market + RSI oversold = mean revert long
        if is_range and rsi_oversold and volume_ok:
            new_signal = POSITION_SIZE
        
        # Regime 2: Range market + RSI EXTREME oversold = strong mean revert (ignore volume)
        elif is_range and rsi_extreme_oversold:
            new_signal = POSITION_SIZE
        
        # Regime 3: Trending market + 4h bull + 12h bull + RSI pullback = trend follow long
        elif is_trend and hma_4h_slope_bull and price_above_hma_4h:
            if hma_12h_slope_bull and rsi_14[i] < 50.0:  # Pullback in uptrend
                new_signal = POSITION_SIZE
        
        # Regime 4: 4h bull + price below 4h HMA (pullback entry) + RSI recovering
        elif hma_4h_slope_bull and price_below_hma_4h and rsi_14[i] > 35.0 and rsi_14[i] < 50.0:
            if volume_ok:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market + RSI overbought = mean revert short
        if new_signal == 0.0:  # Only if no long signal
            if is_range and rsi_overbought and volume_ok:
                new_signal = -POSITION_SIZE
            
            # Regime 2: Range market + RSI EXTREME overbought = strong mean revert
            elif is_range and rsi_extreme_overbought:
                new_signal = -POSITION_SIZE
            
            # Regime 3: Trending market + 4h bear + 12h bear + RSI pullback = trend follow short
            elif is_trend and hma_4h_slope_bear and price_below_hma_4h:
                if hma_12h_slope_bear and rsi_14[i] > 50.0:  # Pullback in downtrend
                    new_signal = -POSITION_SIZE
            
            # Regime 4: 4h bear + price above 4h HMA (pullback entry) + RSI weakening
            elif hma_4h_slope_bear and price_above_hma_4h and rsi_14[i] > 50.0 and rsi_14[i] < 65.0:
                if volume_ok:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON HTF TREND FLIP ===
        if in_position and position_side > 0:
            # Exit long if 4h trend turns bear and price below 4h HMA
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend turns bull and price above 4h HMA
            if hma_4h_slope_bull and price_above_hma_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals