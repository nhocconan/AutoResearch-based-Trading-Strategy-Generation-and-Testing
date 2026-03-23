#!/usr/bin/env python3
"""
Experiment #004: 4h Primary + 12h HTF — Dual Regime with KAMA + RSI + Choppiness

Hypothesis: 4h timeframe with 12h trend bias should generate 20-50 trades/year with
positive Sharpe across all symbols. Key insight: Entry conditions must be LOOSE enough
to guarantee trade generation while regime filtering maintains edge.

Why this should work:
- 4h primary = moderate trade frequency (targets 30-60/year)
- 12h HMA = strong trend filter, avoids counter-trend in strong moves
- Choppiness Index = regime detection (range vs trend) for adaptive logic
- KAMA = adapts to volatility better than fixed EMA
- LOOSE RSI thresholds (30/70) = ensures trade generation
- ATR stoploss = limits drawdown on each position

Key components:
1. 12h HMA(21): Macro trend bias (only trade with HTF trend)
2. 4h Choppiness(14): Regime detection (>55 = range, <45 = trend)
3. 4h KAMA(10,2,30): Adaptive trend following
4. 4h RSI(14): Entry timing with loose thresholds
5. 4h ATR(14): 2.5*ATR trailing stoploss

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_chop_regime_12h_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market efficiency - smooth in trends, responsive in ranges.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER) = |change| / sum of absolute changes
    change = np.abs(close_s.diff())
    sum_change = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    net_change = np.abs(close_s.diff(er_period))
    
    er = np.zeros(n)
    mask = sum_change.values > 0
    er[mask] = net_change.values[mask] / sum_change.values[mask]
    er[:er_period] = 1.0  # Default to trending at start
    
    # Smoothing Constant (SC)
    fast_sc_val = 2.0 / (fast_sc + 1)
    slow_sc_val = 2.0 / (slow_sc + 1)
    sc = er * (fast_sc_val - slow_sc_val) + slow_sc_val
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close_s.iloc[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for macro bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    
    kama_4h = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate KAMA slope (5-bar lookback)
    kama_slope = np.zeros(n)
    for i in range(5, n):
        kama_slope[i] = kama_4h[i] - kama_4h[i-5]
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(kama_4h[i]) or atr_14[i] == 0:
            continue
        
        # === 12H MACRO BIAS ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === CHOPPINESS REGIME (LOOSE thresholds for more trades) ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # LOOSE: was 55, now 50
        is_trending = chop_value < 50.0  # LOOSE: was 45, now 50
        
        # === RSI EXTREMES (LOOSE for trade generation) ===
        rsi_oversold = rsi_14[i] < 40.0  # LOOSE: was 30, now 40 for more longs
        rsi_overbought = rsi_14[i] > 60.0  # LOOSE: was 70, now 60 for more shorts
        rsi_neutral = rsi_14[i] > 45.0 and rsi_14[i] < 55.0
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_4h[i] and kama_slope[i] > 0
        kama_bearish = close[i] < kama_4h[i] and kama_slope[i] < 0
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.015  # Within 1.5% of lower band
        price_near_bb_upper = close[i] > bb_upper[i] * 0.985  # Within 1.5% of upper band
        
        # === VOLATILITY FILTER ===
        vol_elevated = atr_7[i] > atr_14[i] * 1.05  # Recent vol slightly above average
        
        # === ADAPTIVE REGIME ENTRY LOGIC (LOOSE CONDITIONS) ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion ---
        if is_ranging:
            # Long: RSI oversold OR near BB lower + 12h bias helps
            if rsi_oversold or price_near_bb_lower:
                if price_above_hma_12h or vol_elevated:  # Either 12h bullish OR high vol
                    new_signal = POSITION_SIZE
            
            # Short: RSI overbought OR near BB upper + 12h bias helps
            elif rsi_overbought or price_near_bb_upper:
                if price_below_hma_12h or vol_elevated:  # Either 12h bearish OR high vol
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following ---
        elif is_trending:
            # Long: KAMA bullish + RSI not overbought + 12h confirms
            if kama_bullish and rsi_14[i] < 65.0:
                if price_above_hma_12h:  # 12h trend confirmation
                    new_signal = POSITION_SIZE
            
            # Short: KAMA bearish + RSI not oversold + 12h confirms
            elif kama_bearish and rsi_14[i] > 35.0:
                if price_below_hma_12h:  # 12h trend confirmation
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: Simple KAMA crossover if no regime signal ---
        if new_signal == 0.0:
            # Long: Price crosses above KAMA + RSI rising
            if close[i] > kama_4h[i] and close[i-1] <= kama_4h[i-1]:
                if rsi_14[i] > rsi_14[i-1] and rsi_14[i] > 40:
                    new_signal = POSITION_SIZE
            
            # Short: Price crosses below KAMA + RSI falling
            elif close[i] < kama_4h[i] and close[i-1] >= kama_4h[i-1]:
                if rsi_14[i] < rsi_14[i-1] and rsi_14[i] < 60:
                    new_signal = -POSITION_SIZE
        
        # --- ADDITIONAL FALLBACK: BB Mean Reversion ---
        if new_signal == 0.0:
            # Long at BB lower with 12h support
            if price_near_bb_lower and price_above_hma_12h:
                new_signal = POSITION_SIZE
            # Short at BB upper with 12h resistance
            elif price_near_bb_upper and price_below_hma_12h:
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
        # Exit long if 12h trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_12h and kama_bearish:
                new_signal = 0.0
        
        # Exit short if 12h trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_12h and kama_bullish:
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