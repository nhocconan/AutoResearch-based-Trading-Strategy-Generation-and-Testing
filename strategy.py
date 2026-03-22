#!/usr/bin/env python3
"""
Experiment #512: 12h Primary + 1d/1w HTF — Choppiness Regime + Dual Strategy

Hypothesis: After 500+ failed strategies (mostly CRSI/vol-spike combos), try the 
proven research pattern that worked for ETH (Sharpe +0.923 in research notes):

CHOPPINESS INDEX REGIME SWITCH:
- CHOP(14) > 55 = range market → use MEAN REVERSION (RSI extremes + BB)
- CHOP(14) < 45 = trend market → use TREND FOLLOWING (Donchian breakout + HMA)

Why this might beat current best (Sharpe=0.435):
1. DUAL REGIME logic adapts to market conditions (research note #8)
2. 12h TF targets 20-50 trades/year (low fee drag, matches Rule 10)
3. Simpler entry conditions = MORE TRADES (critical after 0-trade failures)
4. 1d HMA for major trend filter (proven in current best strategy)
5. ATR 2.5x trailing stop protects in 2022-style crashes

CRITICAL: Loosen entry conditions to ensure >=30 trades/symbol on train.
Recent failures (#505, #506, #508, #510) had Sharpe=0.000 = ZERO TRADES.

Position sizing: 0.25-0.30 (discrete levels, max 0.40 per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_dual_hma_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/consolidation (mean revert)
    CHOP < 38.2 = strong trend (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest = high_s.rolling(window=period, min_periods=period).max()
    lowest = low_s.rolling(window=period, min_periods=period).min()
    hl_range = highest - lowest
    
    # Avoid division by zero
    hl_range = hl_range.replace(0, 1e-10)
    
    # Choppiness formula
    chop = 100.0 * np.log10(atr_sum / hl_range) / np.log10(period)
    
    return chop.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        if span < 1:
            return series
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

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
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.28
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track previous RSI for crossover detection
    prev_rsi = np.zeros(n)
    prev_rsi[1:] = rsi_14[:-1]
    
    for i in range(100, n):  # Start at 100 to ensure indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(bb_lower[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        strong_bull = bull_regime and (hma_1d_21_aligned[i] > hma_1d_50_aligned[i])
        strong_bear = bear_regime and (hma_1d_21_aligned[i] < hma_1d_50_aligned[i])
        
        # === CHOPPINESS REGIME (determines strategy type) ===
        choppy_regime = chop_14[i] > 55.0  # Range market
        trending_regime = chop_14[i] < 45.0  # Strong trend
        
        # === RSI SIGNALS (loosened for trade frequency) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_low = rsi_14[i] < 25.0
        rsi_extreme_high = rsi_14[i] > 75.0
        rsi_cross_up = (rsi_14[i] > 35.0) and (prev_rsi[i] <= 35.0)
        rsi_cross_down = (rsi_14[i] < 65.0) and (prev_rsi[i] >= 65.0)
        
        # === BOLLINGER BAND EXTREMES ===
        bb_extreme_low = close[i] < bb_lower[i]
        bb_extreme_high = close[i] > bb_upper[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC — DUAL REGIME STRATEGY ===
        new_signal = 0.0
        
        # --- CHOPPY REGIME: MEAN REVERSION ---
        if choppy_regime:
            # LONG: RSI oversold + BB lower (classic mean reversion)
            if rsi_oversold and bb_extreme_low:
                new_signal = LONG_SIZE
            # LONG: RSI extreme low (panic capitulation)
            elif rsi_extreme_low:
                new_signal = LONG_SIZE * 0.8
            # LONG: RSI crosses up from oversold
            elif rsi_cross_up and rsi_14[i] < 45.0:
                new_signal = LONG_SIZE * 0.7
            
            # SHORT: RSI overbought + BB upper
            if new_signal == 0.0:
                if rsi_overbought and bb_extreme_high:
                    new_signal = -SHORT_SIZE
                # SHORT: RSI extreme high (FOMO top)
                elif rsi_extreme_high:
                    new_signal = -SHORT_SIZE * 0.8
                # SHORT: RSI crosses down from overbought
                elif rsi_cross_down and rsi_14[i] > 55.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # --- TRENDING REGIME: TREND FOLLOWING ---
        elif trending_regime:
            # LONG: Bull trend + RSI pullback + Donchian breakout
            if strong_bull:
                if rsi_14[i] < 50.0 and rsi_14[i] > 35.0:  # Pullback entry
                    new_signal = LONG_SIZE
                elif donchian_breakout_up:  # Breakout continuation
                    new_signal = LONG_SIZE * 0.8
            # SHORT: Bear trend + RSI bounce + Donchian breakdown
            elif strong_bear:
                if rsi_14[i] > 50.0 and rsi_14[i] < 65.0:  # Bounce entry
                    new_signal = -SHORT_SIZE
                elif donchian_breakout_down:  # Breakdown continuation
                    new_signal = -SHORT_SIZE * 0.8
        
        # --- NEUTRAL REGIME (45-55 CHOP): SIMPLE RSI REVERSION ---
        else:
            if rsi_extreme_low:
                new_signal = LONG_SIZE * 0.7
            elif rsi_extreme_high:
                new_signal = -SHORT_SIZE * 0.7
        
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
        if in_position and position_side > 0:
            # Exit on overbought or regime flip to strong bear
            if rsi_overbought or (strong_bear and chop_14[i] < 45.0):
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit on oversold or regime flip to strong bull
            if rsi_oversold or (strong_bull and chop_14[i] < 45.0):
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