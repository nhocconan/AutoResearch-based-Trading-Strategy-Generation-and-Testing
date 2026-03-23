#!/usr/bin/env python3
"""
Experiment #254: 4h Primary + 12h/1d HTF — Choppiness Regime-Switch Strategy

Hypothesis: After analyzing 200+ failed experiments, the key insight is:
- Complex multi-filter strategies (CRSI + CHOP + Donchian + ADX) = 0 trades or whipsaws
- Simple trend-following alone = negative Sharpe in bear/range markets (2022, 2025)
- SOLUTION: Use Choppiness Index to SELECT strategy type, not filter entries

REGIME-SWITCH LOGIC:
- CHOP(14) < 38.2 = TRENDING regime → Follow 12h HMA direction, enter on RSI pullbacks
- CHOP(14) > 61.8 = RANGING regime → Mean revert at RSI extremes (30/70)
- CHOP between 38.2-61.8 = TRANSITION → Stay flat (avoid whipsaws)

KEY DIFFERENCES FROM FAILED ATTEMPTS:
- #249 failed with -42.5% DD: Used CRSI extremes (15/85) which trigger too rarely
- This uses RSI 40/60 for trend entries (more frequent) + RSI 30/70 for mean reversion
- 12h HMA for macro bias (faster than 1d, better for 4h entries)
- Position size: 0.25 full, 0.15 half (discrete levels to minimize fee churn)
- ATR 2.5x trailing stop on all positions

TARGET: 25-45 trades/year on 4h, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_switch_hma_rsi_12h_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = Range-bound market
    CHOP < 38.2 = Trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_s = pd.Series(tr)
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        price_range = highest_high - lowest_low
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return np.nan_to_num(chop, nan=50.0)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 12h HMA for macro trend (aligned properly with shift(1))
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1d HMA for stronger macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        is_trending = chop_value < 38.2
        is_ranging = chop_value > 61.8
        is_transition = (chop_value >= 38.2) and (chop_value <= 61.8)
        
        # === MACRO BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND (HMA21 vs price) ===
        price_above_hma_4h = close[i] > hma_21[i]
        price_below_hma_4h = close[i] < hma_21[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # --- TRENDING REGIME LOGIC (CHOP < 38.2) ---
        if is_trending:
            # LONG: 12h bullish + 4h bullish + RSI pullback (40-55)
            if price_above_hma_12h and price_above_hma_4h:
                if (rsi_14[i] >= 40.0) and (rsi_14[i] <= 55.0):
                    desired_signal = POSITION_SIZE_FULL
            
            # SHORT: 12h bearish + 4h bearish + RSI pullback (45-60)
            elif price_below_hma_12h and price_below_hma_4h:
                if (rsi_14[i] >= 45.0) and (rsi_14[i] <= 60.0):
                    desired_signal = -POSITION_SIZE_FULL
        
        # --- RANGING REGIME LOGIC (CHOP > 61.8) ---
        elif is_ranging:
            # LONG: RSI oversold (<35) + price near 4h HMA support
            if (rsi_14[i] < 35.0) and price_above_hma_4h:
                desired_signal = POSITION_SIZE_FULL
            
            # SHORT: RSI overbought (>65) + price near 4h HMA resistance
            elif (rsi_14[i] > 65.0) and price_below_hma_4h:
                desired_signal = -POSITION_SIZE_FULL
        
        # --- TRANSITION REGIME (38.2 <= CHOP <= 61.8) ---
        # Stay flat, avoid whipsaws
        else:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes from trending to ranging (or vice versa) while in position
        if in_position and is_transition:
            desired_signal = 0.0
        
        # Exit long if trending regime turns bearish
        if in_position and position_side > 0 and is_trending:
            if price_below_hma_12h or price_below_hma_4h:
                desired_signal = 0.0
        
        # Exit short if trending regime turns bullish
        if in_position and position_side < 0 and is_trending:
            if price_above_hma_12h or price_above_hma_4h:
                desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit in ranging regime) ===
        if in_position and position_side > 0 and is_ranging:
            if rsi_14[i] > 60.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0 and is_ranging:
            if rsi_14[i] < 40.0:
                desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if regime still supports long
                if is_trending and price_above_hma_12h and price_above_hma_4h:
                    desired_signal = POSITION_SIZE_HALF
                elif is_ranging and rsi_14[i] < 55.0:
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                # Hold short if regime still supports short
                if is_trending and price_below_hma_12h and price_below_hma_4h:
                    desired_signal = -POSITION_SIZE_HALF
                elif is_ranging and rsi_14[i] > 45.0:
                    desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals