#!/usr/bin/env python3
"""
Experiment #1488: 30m Primary + 4h/1d HTF — Loose Multi-TF Trend Following

Hypothesis: After 1110+ failed strategies, the pattern is clear:
1. 30m/1h strategies FAIL when over-filtered (0 trades in #1478, #1480)
2. 1d/12h strategies WORK with simple trend following (#1482 Sharpe=0.237)
3. Best strategy uses: Donchian breakout + HMA trend + RSI pullback

Key insight: For 30m to work, use HTF (1d/4h) for DIRECTION, 30m only for ENTRY TIMING.
Previous 30m attempts failed because they had TOO MANY filters → 0 trades.

This strategy uses:
- 1d HMA(21) for MACRO trend direction (only trade with daily trend)
- 4h HMA(21) for INTERMEDIATE trend confirmation
- 30m Donchian(20) breakout for ENTRY timing (loose: just break + RSI filter)
- RSI(14) loose filter (40-60 range, NOT 45-55) to ensure sufficient trades
- ATR(14)*2.5 trailing stoploss
- Position size: 0.25 (smaller for lower TF to reduce fee drag)

Why this should work on 30m:
1. 1d/4h filters ensure we only trade WITH macro trend (reduces whipsaws)
2. Loose RSI (40-60) ensures we get 30-80 trades/year (not 0 like #1478)
3. Donchian breakout is proven to work (best strategy uses it)
4. Small position size (0.25) limits drawdown on lower TF volatility
5. Discrete signal levels (0.0, ±0.25) minimize fee churn

Timeframe: 30m
HTF: 4h, 1d (call get_htf_data ONCE before loop!)
Position Size: 0.25
Target: 40-80 trades/year, Sharpe > 0.3, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_donchian_rsi_4h1d_loose_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(data, w_period):
        out = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1)
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            out[i] = np.sum(data[i - w_period + 1:i + 1] * weights) / np.sum(weights)
        return out
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    # WMA of diff with sqrt(n) period
    hma = wma(diff, sqrt_n)
    
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
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1 - CRITICAL) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === CALCULATE AND ALIGN HTF INDICATORS ===
    # 1d HMA for MACRO trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 4h HMA for INTERMEDIATE trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # === CALCULATE PRIMARY (30m) INDICATORS ===
    hma_30m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for lower TF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # === SKIP IF INDICATORS NOT READY ===
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_30m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        h4_bull = close[i] > hma_4h_aligned[i]
        h4_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (30m HMA) ===
        h30_bull = close[i] > hma_30m[i]
        h30_bear = close[i] < hma_30m[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI MOMENTUM - LOOSE bands for more trades (40-60) ===
        # This is KEY to avoid 0 trades like #1478, #1480
        rsi_bullish = rsi[i] > 40.0
        rsi_bearish = rsi[i] < 60.0
        rsi_neutral = 40.0 <= rsi[i] <= 60.0
        
        # === SMA 50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === DESIRED SIGNAL - LOOSE TREND FOLLOWING ===
        desired_signal = 0.0
        
        # LONG: All HTF bullish + breakout or pullback entry
        # Condition 1: Strong breakout with all trends aligned
        if daily_bull and h4_bull and h30_bull:
            if breakout_high and rsi_bullish:
                desired_signal = BASE_SIZE  # Full size on strong breakout
            # Condition 2: Pullback entry (price near HMA but RSI still ok)
            elif close[i] > hma_30m[i] * 0.99 and rsi[i] > 45.0 and above_sma50:
                desired_signal = BASE_SIZE * 0.8  # Slightly smaller on pullback
            # Condition 3: Weakest long signal (just trend aligned)
            elif h30_bull and rsi[i] > 50.0:
                desired_signal = BASE_SIZE * 0.6
        
        # SHORT: All HTF bearish + breakout or pullback entry
        elif daily_bear and h4_bear and h30_bear:
            if breakout_low and rsi_bearish:
                desired_signal = -BASE_SIZE  # Full size on strong breakout
            # Condition 2: Pullback entry
            elif close[i] < hma_30m[i] * 1.01 and rsi[i] < 55.0 and below_sma50:
                desired_signal = -BASE_SIZE * 0.8
            # Condition 3: Weakest short signal
            elif h30_bear and rsi[i] < 50.0:
                desired_signal = -BASE_SIZE * 0.6
        
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
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.6
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