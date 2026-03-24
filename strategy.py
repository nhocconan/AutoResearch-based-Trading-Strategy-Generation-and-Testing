#!/usr/bin/env python3
"""
Experiment #1484: 4h Primary + 12h HTF — Simplified HMA Trend + Donchian Breakout

Hypothesis: After analyzing 1100+ failed strategies, the pattern is clear:
1. 4h strategies fail when too complex (regime-switching, too many filters)
2. 12h/1d strategies work better with simpler logic (exp #1477, #1482 kept)
3. KAMA is too slow/complex — HMA is more responsive for 4h entries
4. Donchian breakout + HMA trend is the most proven combination

Key insight from #1482 (12h, Sharpe=0.237): Simple trend-follow with HTF filter works.
This strategy uses:
- 12h HMA(21) for macro trend direction (call get_htf_data ONCE!)
- 4h HMA(21) for local trend confirmation
- Donchian(20) breakout for entry trigger (proven in #1477)
- RSI(14) loose filter (45-55) for momentum confirmation
- ATR(14)*2.5 trailing stoploss

Why 4h + 12h should work:
1. 4h = target 20-50 trades/year (minimal fee drag ~1-2.5%)
2. 12h HMA filter prevents trading against macro trend
3. HMA is more responsive than KAMA/EMA for 4h entries
4. Donchian breakout ensures we catch trending moves
5. Loose RSI filter ensures sufficient trades (not over-filtered)
6. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Timeframe: 4h
HTF: 12h (call get_htf_data ONCE before loop!)
Position Size: 0.28 (discrete levels)
Target: 25-50 trades/year, Sharpe > 0.25, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_rsi_12h_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — more responsive than EMA/KAMA
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Convert to pandas for WMA calculation
    close_series = pd.Series(close)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close_series, half_period)
    wma_full = wma(close_series, period)
    
    # HMA formula
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    return hma.values

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
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
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(50, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (12h HMA) - direction bias ===
        # Only trade in direction of 12h trend
        daily_bull = close[i] > hma_12h_aligned[i]
        daily_bear = close[i] < hma_12h_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        # Breakout above previous high or below previous low
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI MOMENTUM - LOOSE bands for more trades (45-55) ===
        rsi_bullish = rsi[i] > 45.0
        rsi_bearish = rsi[i] < 55.0
        rsi_neutral = 45.0 <= rsi[i] <= 55.0
        
        # === DESIRED SIGNAL — SIMPLIFIED TREND FOLLOWING ===
        desired_signal = 0.0
        
        # LONG: 12h bull + 4h bull + Breakout + RSI support
        if daily_bull and hma_bull:
            if breakout_high and rsi_bullish:
                desired_signal = BASE_SIZE
            elif hma_bull and rsi[i] > 50.0 and close[i] > hma_12h_aligned[i]:
                desired_signal = BASE_SIZE * 0.7
        
        # SHORT: 12h bear + 4h bear + Breakout + RSI support
        elif daily_bear and hma_bear:
            if breakout_low and rsi_bearish:
                desired_signal = -BASE_SIZE
            elif hma_bear and rsi[i] < 50.0 and close[i] < hma_12h_aligned[i]:
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.7
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
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals