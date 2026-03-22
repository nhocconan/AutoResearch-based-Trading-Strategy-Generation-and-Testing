#!/usr/bin/env python3
"""
Experiment #516: 12h Primary + 1d HTF — Dual Regime (Trend/Mean Revert) + HMA + RSI

Hypothesis: After 462 failed strategies (mostly over-filtered), simplify drastically.
Key insight from failures: Too many confluence conditions = 0 trades (Sharpe=0.000).

This strategy uses a DUAL REGIME approach:
1. TREND REGIME (ADX > 20): Follow 1d HMA direction, enter on 12h RSI pullbacks
2. RANGE REGIME (ADX < 20): Mean revert at Bollinger Band extremes

Why this might work:
- Fewer conflicting filters = MORE trades (critical: need >=30/symbol on train)
- 12h TF targets 20-50 trades/year (lower fee drag, matches Rule 10)
- 1d HMA provides major trend filter without over-complication
- ADX regime switch is proven (research note #2: Choppiness/ADX meta-filter)
- Simple RSI thresholds (30/70) trigger frequently enough

Position sizing: 0.25-0.30 discrete levels (Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_hma_rsi_1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === ADX REGIME DETECTION ===
        trend_regime = adx_14[i] > 20  # Trending market
        range_regime = adx_14[i] <= 20  # Ranging market
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_low = rsi_14[i] < 25.0
        rsi_extreme_high = rsi_14[i] > 75.0
        
        # === BOLLINGER BAND EXTREMES ===
        bb_extreme_low = close[i] < bb_lower[i]
        bb_extreme_high = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # TREND REGIME: Follow 1d trend, enter on pullbacks
        if trend_regime:
            # Long: Bull regime + RSI pullback (not extreme, just dipped)
            if bull_regime and rsi_oversold and not rsi_extreme_low:
                new_signal = LONG_SIZE
            # Long: Strong bull + RSI extreme (capitulation entry)
            elif bull_regime and hma_slope_bull and rsi_extreme_low:
                new_signal = LONG_SIZE
            # Short: Bear regime + RSI bounce (not extreme)
            elif bear_regime and rsi_overbought and not rsi_extreme_high:
                new_signal = -SHORT_SIZE
            # Short: Strong bear + RSI extreme (FOMO top)
            elif bear_regime and hma_slope_bear and rsi_extreme_high:
                new_signal = -SHORT_SIZE
        
        # RANGE REGIME: Mean revert at BB extremes
        if range_regime and new_signal == 0.0:
            # Long: Price at lower BB + RSI oversold
            if bb_extreme_low and rsi_oversold:
                new_signal = LONG_SIZE
            # Short: Price at upper BB + RSI overbought
            elif bb_extreme_high and rsi_overbought:
                new_signal = -SHORT_SIZE
            # Long: Extreme BB touch (stronger signal)
            elif bb_extreme_low and rsi_extreme_low:
                new_signal = LONG_SIZE
            # Short: Extreme BB touch (stronger signal)
            elif bb_extreme_high and rsi_extreme_high:
                new_signal = -SHORT_SIZE
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme) ===
        # Exit long if regime flips strongly bearish
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear and adx_14[i] > 25:
                new_signal = 0.0
            # Exit on RSI overbought in range regime
            if range_regime and rsi_overbought:
                new_signal = 0.0
        
        # Exit short if regime flips strongly bullish
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull and adx_14[i] > 25:
                new_signal = 0.0
            # Exit on RSI oversold in range regime
            if range_regime and rsi_oversold:
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
                # Flip position
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