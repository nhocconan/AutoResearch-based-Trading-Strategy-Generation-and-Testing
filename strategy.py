#!/usr/bin/env python3
"""
Experiment #042: 12h Primary + 1d/1w HTF — Fisher Transform + KAMA Trend + Choppiness Regime

Hypothesis: Based on research showing Ehlers Fisher Transform catches reversals in bear rallies
and KAMA adapts to volatility better than EMA/HMA, I'm combining these with Choppiness Index
regime detection at 12h timeframe.

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, enters at extremes (-1.5/+1.5) — proven reversal catcher
2. KAMA TREND: Kaufman Adaptive MA adapts to volatility, less whipsaw than HMA/EMA
3. CHOPPINESS REGIME: CHOP(14) > 61.8 = range (mean revert), < 38.2 = trend (trend follow)
4. 1d/1w HMA for macro bias — only enter longs when 1w HMA bullish
5. ASYMMETRIC entries: with macro trend = 1 filter, against = 3 filters

Why 12h works:
- Targets 20-50 trades/year (Rule 10 — fee efficient)
- #032 12h Donchian+CRSI got Sharpe=0.419 — proven TF
- Less noise than 4h, more signals than 1d

Entry conditions (LOOSE enough to generate 10+ trades/symbol):
- Long mean-revert: Fisher < -1.5 + CHOP > 55 + price > 1w HMA
- Short mean-revert: Fisher > +1.5 + CHOP > 55 + price < 1w HMA
- Long trend: Fisher crosses up from <-1 + CHOP < 45 + KAMA bullish + 1d HMA bullish
- Short trend: Fisher crosses down from >+1 + CHOP < 45 + KAMA bearish + 1d HMA bearish

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_chop_regime_1d1w_v1"
timeframe = "12h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise — fast in trends, slow in ranges.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Price change over er_period
    price_change = np.abs(close_s - close_s.shift(er_period)).values
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        if volatility[i] > 0:
            er[i] = price_change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer signals.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        price_norm = ((high[i] + low[i]) / 2.0 - ll) / range_val
        price_norm = np.clip(price_norm, 0.001, 0.999)  # Avoid log(0)
        
        # Fisher calculation
        fisher[i] = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(kama_12h[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Mean reversion regime
        is_trending = chop_value < 45.0  # Trend following regime
        
        # === MACRO TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_slope_bull = kama_12h[i] > kama_12h[i-5] if i >= 5 else False
        kama_slope_bear = kama_12h[i] < kama_12h[i-5] if i >= 5 else False
        price_above_kama = close[i] > kama_12h[i]
        price_below_kama = close[i] < kama_12h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = (fisher_signal[i-1] < -1.0) and (fisher[i] > fisher_signal[i])
        fisher_cross_down = (fisher_signal[i-1] > 1.0) and (fisher[i] < fisher_signal[i])
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion (Fisher extremes) ---
        if is_ranging:
            # Long: Fisher oversold + price above macro HMA (easier entry)
            if fisher_oversold:
                if price_above_hma_1w:  # With macro trend = 1 filter
                    new_signal = POSITION_SIZE
                elif price_above_hma_1d and price_above_kama:  # Against = need confluence
                    new_signal = POSITION_SIZE
            
            # Short: Fisher overbought + price below macro HMA
            if fisher_overbought and new_signal == 0.0:
                if price_below_hma_1w:  # With macro trend = 1 filter
                    new_signal = -POSITION_SIZE
                elif price_below_hma_1d and price_below_kama:  # Against = need confluence
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following (Fisher cross + KAMA) ---
        elif is_trending:
            # Long: Fisher crosses up + KAMA bullish + 1d HMA bullish
            if fisher_cross_up and kama_slope_bull and price_above_kama:
                if price_above_hma_1d:  # Require 1d confirmation
                    new_signal = POSITION_SIZE
            
            # Short: Fisher crosses down + KAMA bearish + 1d HMA bearish
            if fisher_cross_down and kama_slope_bear and price_below_kama:
                if price_below_hma_1d:  # Require 1d confirmation
                    if new_signal == 0.0:  # Don't override long signal
                        new_signal = -POSITION_SIZE
        
        # --- DONCHIAN BREAKOUT (additional entry in trending regime) ---
        if is_trending and new_signal == 0.0:
            if donchian_breakout_up and price_above_hma_1d and price_above_hma_1w:
                new_signal = POSITION_SIZE
            elif donchian_breakout_down and price_below_hma_1d and price_below_hma_1w:
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime changes from ranging to strongly trending bearish
        if in_position and position_side > 0:
            if is_trending and kama_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if regime changes from ranging to strongly trending bullish
        if in_position and position_side < 0:
            if is_trending and kama_slope_bull and price_above_hma_1d:
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