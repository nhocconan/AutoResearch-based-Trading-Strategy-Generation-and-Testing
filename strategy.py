#!/usr/bin/env python3
"""
Experiment #1527: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: Based on proven patterns from research for daily timeframe:
1. 1w HMA(21) provides macro trend bias (HTF filter) — only trade in direction of weekly trend
2. 1d Donchian(20) breakout captures momentum moves (proven on SOL with Sharpe +0.782)
3. 1d HMA(16) confirms trend direction (reduces false breakouts)
4. 1d RSI(14) loose filter ensures entries (RSI < 55 for longs, > 45 for shorts)
5. ATR(14) 2.5x trailing stop for risk management
6. Position size 0.30 with discrete levels to minimize fee churn
7. Target: 20-50 trades/year on 1d timeframe (natural frequency)

Key insights from 1100+ failed strategies:
- 1d timeframe works best for trend strategies (current best Sharpe=0.618 is 1d)
- HTF (1w) trend filter improves Sharpe by avoiding counter-trend trades
- Donchian breakout + HMA confirmation = fewer false signals
- Loose RSI filter ensures we get trades (not too strict like RSI < 30)
- Simple is better: complex filters = 0 trades (#1515, #1518 had Sharpe=0.000)

Design:
- 1w HMA(21) for macro trend bias (HTF filter) — call ONCE before loop
- 1d HMA(16) for primary trend confirmation
- 1d Donchian(20) for breakout detection
- 1d RSI(14) for entry timing (loose: < 55 for longs, > 45 for shorts)
- 1d ATR(14) 2.5x trailing stop
- Position size 0.30 (discrete: 0.0, ±0.20, ±0.30)
- Target: 80-200 trades/train (4 years), 20-50 trades/test (15 months)

Timeframe: 1d (as required by experiment)
HTF: 1w (weekly trend bias)
Position Size: 0.30 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_atr_v2"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

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
    """Donchian Channel - highest high and lowest low over period"""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=16)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 1d (20-50 trades/year target)
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
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
        
        # === MACRO TREND (1w HMA) - primary direction bias ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA) ===
        hma_bull = close[i] > hma_1d[i]
        hma_bear = close[i] < hma_1d[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 1e-10:
            donchian_position = (close[i] - donchian_lower[i]) / donchian_range
        else:
            donchian_position = 0.5
        
        # Breakout signals (price near channel bounds)
        donchian_breakout_long = donchian_position > 0.75  # Price in upper 25%
        donchian_breakout_short = donchian_position < 0.25  # Price in lower 25%
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_not_overbought = rsi[i] < 60.0  # Loose for longs
        rsi_not_oversold = rsi[i] > 40.0  # Loose for shorts
        rsi_mild_oversold = rsi[i] < 50.0  # For long entries
        rsi_mild_overbought = rsi[i] > 50.0  # For short entries
        
        # === DESIRED SIGNAL - DONCHIAN + HMA + RSI ===
        desired_signal = 0.0
        
        # LONG SIGNALS (only when weekly trend is bullish)
        if weekly_bull:
            # Primary: Donchian breakout + HMA bull + RSI not overbought
            if donchian_breakout_long and hma_bull and rsi_not_overbought:
                desired_signal = BASE_SIZE
            # Secondary: HMA bull + RSI mild oversold (pullback entry)
            elif hma_bull and rsi_mild_oversold:
                desired_signal = BASE_SIZE * 0.7
            # Fallback: Weekly bull + HMA bull (simplest, ensures trades)
            elif weekly_bull and hma_bull:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT SIGNALS (only when weekly trend is bearish)
        elif weekly_bear:
            # Primary: Donchian breakdown + HMA bear + RSI not oversold
            if donchian_breakout_short and hma_bear and rsi_not_oversold:
                desired_signal = -BASE_SIZE
            # Secondary: HMA bear + RSI mild overbought (pullback entry)
            elif hma_bear and rsi_mild_overbought:
                desired_signal = -BASE_SIZE * 0.7
            # Fallback: Weekly bear + HMA bear (simplest, ensures trades)
            elif weekly_bear and hma_bear:
                desired_signal = -BASE_SIZE * 0.5
        
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
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.4:
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