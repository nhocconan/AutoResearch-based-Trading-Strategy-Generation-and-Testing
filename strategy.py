#!/usr/bin/env python3
"""
Experiment #265: 1h Primary + 4h/1d HTF — Trend Pullback with Choppiness Regime

Hypothesis: After 224 failed experiments, the pattern is clear:
- OVER-FILTERING causes 0-trade scenarios (#255, #260 had Sharpe=0.000)
- Complex regime-switching (CHOP + CRSI + Donchian) creates whipsaws
- Volume/session filters kill too many valid signals

SOLUTION: Simpler approach that WORKED in #251 template:
- 4h HMA(21) for trend direction (faster than 1d for 1h entries)
- 1h RSI(14) pullback to 40-60 zone (NOT extreme 30/70 - triggers more often)
- Choppiness Index(14) as META-filter: CHOP < 50 = trending (trade), CHOP > 60 = range (skip)
- ATR(14) 3.0x trailing stoploss (wider for 1h volatility)
- Position size: 0.25 full, 0.15 half (conservative for 1h frequency)

KEY INSIGHT: 1h needs MORE trades than 4h (target 40-80/year) but still filtered by HTF.
RSI 40-60 triggers ~35% of bars vs RSI 30/70 at ~10%. CHOP filter prevents range whipsaws.

TARGET: Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL), trades >= 40/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_regime_4h_atr_v1"
timeframe = "1h"
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
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    We use threshold 50-60 for meta-filter.
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_series = pd.Series(tr)
    atr_sum = tr_series.rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        price_range = highest_high - lowest_low
        price_range = np.maximum(price_range, 1e-10)  # avoid div by zero
        chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1h indicators (primary timeframe)
    hma_21_1h = calculate_hma(close, 21)
    atr_14_1h = calculate_atr(high, low, close, period=14)
    rsi_14_1h = calculate_rsi(close, period=14)
    chop_14_1h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 4h HMA for trend bias (aligned properly with shift(1))
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.25
    POSITION_SIZE_HALF = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14_1h[i]) or atr_14_1h[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21_1h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14_1h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop_14_1h[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER (Choppiness Index) ===
        # CHOP < 50 = trending regime (trade), CHOP > 60 = choppy (skip)
        is_trending = chop_14_1h[i] < 55.0
        is_choppy = chop_14_1h[i] > 60.0
        
        # === MACRO BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h TREND (HMA slope) ===
        # Use price vs HMA instead of HMA crossover (faster signal)
        hma_bullish_1h = close[i] > hma_21_1h[i]
        hma_bearish_1h = close[i] < hma_21_1h[i]
        
        # === RSI PULLBACK SIGNALS (moderate thresholds for more trades) ===
        # Long: bullish trend + RSI pullback to 40-55 zone
        rsi_pullback_long = (rsi_14_1h[i] >= 40.0) and (rsi_14_1h[i] <= 55.0)
        # Short: bearish trend + RSI pullback to 45-60 zone
        rsi_pullback_short = (rsi_14_1h[i] >= 45.0) and (rsi_14_1h[i] <= 60.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bullish + 1h bullish + RSI pullback + trending regime
        if is_trending and price_above_hma_4h and hma_bullish_1h and rsi_pullback_long:
            desired_signal = POSITION_SIZE_FULL
        
        # SHORT ENTRY: 4h bearish + 1h bearish + RSI pullback + trending regime
        elif is_trending and price_below_hma_4h and hma_bearish_1h and rsi_pullback_short:
            desired_signal = -POSITION_SIZE_FULL
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14_1h[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14_1h[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and price_below_hma_4h:
            desired_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and price_above_hma_4h:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        # Exit long if RSI becomes overbought (>70)
        if in_position and position_side > 0 and rsi_14_1h[i] > 70.0:
            desired_signal = 0.0
        
        # Exit short if RSI becomes oversold (<30)
        if in_position and position_side < 0 and rsi_14_1h[i] < 30.0:
            desired_signal = 0.0
        
        # === CHOPPY REGIME EXIT ===
        # Exit all positions if market becomes choppy
        if in_position and is_choppy:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        # Only hold if we're in position AND no exit signal triggered
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish
                if price_above_hma_4h and hma_bullish_1h and not is_choppy:
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                # Hold short if trend still bearish
                if price_below_hma_4h and hma_bearish_1h and not is_choppy:
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