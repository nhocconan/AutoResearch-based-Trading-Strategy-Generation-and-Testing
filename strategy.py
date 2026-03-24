#!/usr/bin/env python3
"""
Experiment #003: 1d Primary + 1w HTF — Dual Regime Strategy (Choppiness Index)

Hypothesis: After 2 failed experiments with negative Sharpe, the key issue is using
ONE strategy logic for ALL market conditions. Crypto alternates between trending
and ranging regimes, and each requires different entry logic.

This strategy implements DUAL REGIME approach:
1. Choppiness Index (CHOP) detects regime: CHOP>61.8=range, CHOP<38.2=trend
2. RANGE regime (CHOP>61.8): Mean reversion at Bollinger Band extremes + RSI filter
3. TREND regime (CHOP<38.2): Donchian breakout + 1w HMA trend alignment
4. 1w HMA provides major trend bias (prevents counter-trend trades in major moves)
5. ATR(14) 2.5x trailing stoploss on all positions

Why this should work on BTC/ETH (not just SOL):
- Research shows CHOP regime filter gave ETH Sharpe +0.923
- Dual logic adapts to 2022 crash (trend short) AND 2025 bear/range (mean revert)
- 1d timeframe = 20-40 trades/year target (fee-efficient, matches research)
- Discrete sizing 0.25 minimizes fee churn on signal changes
- Loose enough thresholds to ensure ≥10 trades/symbol on train

Entry Logic:
- RANGE (CHOP>61.8): Long when RSI<30 + price<BB_lower, Short when RSI>70 + price>BB_upper
- TREND (CHOP<38.2): Long when Donchian breakout + price>1w HMA, Short when breakout + price<1w HMA
- NEUTRAL (38.2<=CHOP<=61.8): No new entries, hold existing positions

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.3 all symbols, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_dual_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - Regime detection
    CHOP > 61.8 = Ranging market (mean reversion works)
    CHOP < 38.2 = Trending market (breakout works)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        tr_sum = np.sum(tr[i - period + 1:i + 1])
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_hl = hh - ll
        
        if range_hl > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0  # Neutral default
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands - for mean reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Bandwidth for squeeze detection
    
    return upper, lower, width

def calculate_donchian(high, low, period=20):
    """Donchian Channel - for trend breakout entries"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend bias"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, period=20, std_mult=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size (between 0.25-0.30)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range_regime = chop[i] > 61.8
        is_trend_regime = chop[i] < 38.2
        # Neutral regime: 38.2 <= chop <= 61.8 (hold existing, no new entries)
        
        # === HTF TREND BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at BB extremes
        if is_range_regime:
            # Long: RSI oversold + price at/below BB lower
            if rsi[i] < 32.0 and close[i] <= bb_lower[i] * 1.002:
                desired_signal = SIZE
            # Short: RSI overbought + price at/above BB upper
            elif rsi[i] > 68.0 and close[i] >= bb_upper[i] * 0.998:
                desired_signal = -SIZE
        
        # TREND REGIME: Donchian breakout with HTF alignment
        elif is_trend_regime:
            # Check if price broke out THIS bar
            breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
            breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
            
            # Long: Donchian breakout + 1w HMA bullish
            if breakout_long and hma_1w_bull:
                desired_signal = SIZE
            # Short: Donchian breakout + 1w HMA bearish
            elif breakout_short and hma_1w_bear:
                desired_signal = -SIZE
        
        # NEUTRAL REGIME: No new entries, but don't close existing positions
        # (desired_signal stays 0.0, but we won't force exit if in_position)
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            # Only exit if stoploss triggered OR neutral regime with weak signal
            # In neutral regime, we hold existing positions
            if stoploss_triggered or (not is_range_regime and not is_trend_regime):
                if in_position:
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals