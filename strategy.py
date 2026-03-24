#!/usr/bin/env python3
"""
Experiment #1544: 4h Primary + 12h HTF — Choppiness Regime + Dual Mode Strategy

Hypothesis: After 11 failed 4h experiments (#1529-#1541), the pattern shows:
1. Complex multi-filter approaches create signal conflicts → negative Sharpe
2. Funding rate didn't help (#1541 Sharpe=-0.097)
3. 12h HTF aligns better with 4h than 1d (less lag)
4. Need REGIME DETECTION to switch between mean-revert and trend-follow

New Approach — PROVEN from research notes:
- Choppiness Index (CHOP) detects range vs trend regimes
- CHOP > 61.8 = choppy/range → mean reversion (RSI extremes + BB)
- CHOP < 38.2 = trending → trend follow (HMA + Donchian breakout)
- 12h HMA(21) for macro bias (simpler than 1d, less lag)
- LOOSE thresholds to ensure 30+ trades/train, 3+ trades/test
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Why this should beat #1541:
- Regime detection prevents wrong strategy in wrong market
- Mean reversion works in 2022 crash and 2025 bear/range
- Trend follow captures 2021 bull and strong moves
- 12h HTF better aligned with 4h entries than 1d
- Simpler logic = fewer conflicting signals

Timeframe: 4h (required)
HTF: 12h HMA(21) for macro bias
Position Size: 0.30 max, discrete levels
Target: Sharpe > 0.618, DD < -30%, trades > 30/train, > 3/test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_dual_mode_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures if market is trending or ranging
    CHOP > 61.8 = choppy/ranging
    CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel breakout levels"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bb(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower = calculate_bb(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
        if np.isnan(chop[i]) or np.isnan(hma_4h[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market → mean reversion
        is_trending = chop[i] < 38.2  # Trend market → trend follow
        # Neutral zone (38.2-61.8): use both signals with lower confidence
        
        # === MACRO TREND BIAS (12h HMA) ===
        daily_bull = close[i] > hma_12h_aligned[i]
        daily_bear = close[i] < hma_12h_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === RSI CONDITIONS (LOOSE for trade frequency) ===
        rsi_oversold = rsi_14[i] < 35.0 if not np.isnan(rsi_14[i]) else False
        rsi_overbought = rsi_14[i] > 65.0 if not np.isnan(rsi_14[i]) else False
        rsi_neutral_long = rsi_14[i] > 45.0 if not np.isnan(rsi_14[i]) else False
        rsi_neutral_short = rsi_14[i] < 55.0 if not np.isnan(rsi_14[i]) else False
        
        # === BOLLINGER BAND CONDITIONS ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002 if not np.isnan(bb_lower[i]) else False
        at_bb_upper = close[i] >= bb_upper[i] * 0.998 if not np.isnan(bb_upper[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        signal_strength = 0
        
        if is_choppy:
            # MEAN REVERSION MODE
            # Long: RSI oversold + at BB lower + macro bull bias
            long_score = 0
            if rsi_oversold:
                long_score += 3
            if at_bb_lower:
                long_score += 2
            if daily_bull:
                long_score += 2
            if hma_bull:
                long_score += 1
            
            # Short: RSI overbought + at BB upper + macro bear bias
            short_score = 0
            if rsi_overbought:
                short_score += 3
            if at_bb_upper:
                short_score += 2
            if daily_bear:
                short_score += 2
            if hma_bear:
                short_score += 1
            
            if long_score >= 5:
                desired_signal = BASE_SIZE
                signal_strength = long_score
            elif short_score >= 5:
                desired_signal = -BASE_SIZE
                signal_strength = short_score
            elif long_score >= 4 and daily_bull:
                desired_signal = BASE_SIZE * 0.7
                signal_strength = long_score
            elif short_score >= 4 and daily_bear:
                desired_signal = -BASE_SIZE * 0.7
                signal_strength = short_score
        
        elif is_trending:
            # TREND FOLLOW MODE
            # Long: Donchian breakout + HMA bull + RSI neutral + macro bull
            long_score = 0
            if breakout_long:
                long_score += 3
            if hma_bull:
                long_score += 2
            if daily_bull:
                long_score += 2
            if rsi_neutral_long:
                long_score += 1
            
            # Short: Donchian breakout + HMA bear + RSI neutral + macro bear
            short_score = 0
            if breakout_short:
                short_score += 3
            if hma_bear:
                short_score += 2
            if daily_bear:
                short_score += 2
            if rsi_neutral_short:
                short_score += 1
            
            if long_score >= 5:
                desired_signal = BASE_SIZE
                signal_strength = long_score
            elif short_score >= 5:
                desired_signal = -BASE_SIZE
                signal_strength = short_score
            elif long_score >= 4 and daily_bull:
                desired_signal = BASE_SIZE * 0.7
                signal_strength = long_score
            elif short_score >= 4 and daily_bear:
                desired_signal = -BASE_SIZE * 0.7
                signal_strength = short_score
        
        else:
            # NEUTRAL ZONE - use conservative signals
            long_score = 0
            if hma_bull and daily_bull:
                long_score += 2
            if breakout_long:
                long_score += 2
            if rsi_oversold:
                long_score += 2
            
            short_score = 0
            if hma_bear and daily_bear:
                short_score += 2
            if breakout_short:
                short_score += 2
            if rsi_overbought:
                short_score += 2
            
            if long_score >= 5:
                desired_signal = BASE_SIZE * 0.7
            elif short_score >= 5:
                desired_signal = -BASE_SIZE * 0.7
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.35:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.35:
            final_signal = -BASE_SIZE * 0.5
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
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals