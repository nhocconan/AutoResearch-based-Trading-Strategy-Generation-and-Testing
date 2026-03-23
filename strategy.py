#!/usr/bin/env python3
"""
Experiment #741: 4h Primary + 1d HTF — Dual Regime (Trend/Mean-Revert) + ADX Switch

Hypothesis: After analyzing 495+ failed strategies, clear patterns emerge:
1. Complex regime (Chop+CRSI) = 0 trades or negative Sharpe (#729-735)
2. Simple trend (KAMA+Donchian+ADX) on 1d got Sharpe=0.234 (#737) — works
3. Current 4h attempt (#739) got Sharpe=0.012 — too many conflicting entry paths
4. BEST overall uses 4h triple regime with Sharpe=0.612 — needs simpler logic

NEW APPROACH:
1. ADX regime switch: ADX>25 = trend follow, ADX<20 = mean revert (with hysteresis)
2. Trend regime: 1d HMA(21) bias + 4h Donchian(20) breakout entries
3. Mean-revert regime: 4h Bollinger(20,2.5) extremes + RSI(14) confirmation
4. Single clear entry path per regime (not 8 conflicting paths like #739)
5. Simple hold logic: stay in position while regime intact
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete signals: 0.0, ±0.25, ±0.30 to minimize fee churn

Key differences from #739:
- Only 2 entry paths total (1 long, 1 short) instead of 8
- ADX hysteresis (enter 25, exit 20) prevents regime flip-flop
- Bollinger mean-revert for range markets (proven in bear/range)
- Simpler hold logic reduces premature exits
- Target: 25-45 trades/year on 4h timeframe

Timeframe: 4h (primary) + 1d (HTF trend bias)
Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_adx_donchian_bb_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.5):
    """Bollinger Bands for mean reversion detection."""
    n = len(close)
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di / (atr + 1e-10)
        minus_di = 100 * minus_di / (atr + 1e-10)
        di_sum = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / (di_sum + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
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
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.5)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Regime tracking with hysteresis
    in_trend_regime = False  # Track current regime state
    prev_adx = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        # Enter trend regime when ADX > 25, exit when ADX < 20
        current_adx = adx_4h[i]
        
        if not in_trend_regime and current_adx > 25:
            in_trend_regime = True
        elif in_trend_regime and current_adx < 20:
            in_trend_regime = False
        
        prev_adx = current_adx
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        desired_signal = 0.0
        
        # === TREND REGIME (ADX > 20-25) ===
        if in_trend_regime:
            # LONG: Donchian breakout + 1d bullish trend
            if close[i] > donch_upper[i-1] and trend_1d_bullish:
                desired_signal = BASE_SIZE
            
            # SHORT: Donchian breakdown + 1d bearish trend
            elif close[i] < donch_lower[i-1] and trend_1d_bearish:
                desired_signal = -BASE_SIZE
        
        # === MEAN REVERT REGIME (ADX < 20) ===
        else:
            # LONG: Price at BB lower + RSI oversold + 1d bullish bias
            if close[i] <= bb_lower[i] and rsi_4h[i] < 35 and trend_1d_bullish:
                desired_signal = BASE_SIZE
            
            # SHORT: Price at BB upper + RSI overbought + 1d bearish bias
            elif close[i] >= bb_upper[i] and rsi_4h[i] > 65 and trend_1d_bearish:
                desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if regime supports long or neutral
                if (in_trend_regime and trend_1d_bullish) or (not in_trend_regime and trend_1d_bullish):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if regime supports short or neutral
                if (in_trend_regime and trend_1d_bearish) or (not in_trend_regime and trend_1d_bearish):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS — Regime reversal ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses against position
            if trend_1d_bearish and in_trend_regime:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses against position
            if trend_1d_bullish and in_trend_regime:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
        
        signals[i] = desired_signal
    
    return signals