#!/usr/bin/env python3
"""
Experiment #950: 1h Primary + 4h/12h HTF — Simplified Trend + RSI Pullback

Hypothesis: After 679 failed strategies, the key issue is OVER-FILTERING.
Strategies with too many confluence requirements (CHOP + CRSI + Funding + Volume + Session)
generate 0 trades because conditions rarely align.

New approach: SIMPLER but effective
1. 4h HMA(21) = primary trend direction (long only when price > HMA)
2. 12h HMA(21) = macro bias filter (avoid counter-trend trades)
3. 1h RSI(14) = entry timing (pullback entries in direction of HTF trend)
4. 1h ATR(14) = trailing stoploss (2.5x ATR)
5. Volume filter = confirm real moves (>0.8x 20-bar avg)

Key differences from failed strategies:
- NO Choppiness Index (causes 0 trades in #941, #944, #945, #947, #949)
- NO Funding rate (inconsistent across symbols)
- NO CRSI (too strict, caused 0 trades in #938, #942, #946, #948)
- RELAXED RSI thresholds: <40/>60 (not <30/>70)
- HOLD LOGIC: maintain position through minor pullbacks (signal doesn't flip to 0)

Trade frequency target: 40-60 trades/year on 1h timeframe
Position sizing: 0.25 base, 0.15 reduced (discrete levels to minimize fee churn)

Why this should work:
- HTF trend filter prevents whipsaw trades in 2022 crash
- RSI pullback entries catch dips in uptrend / rallies in downtrend
- Simple logic = more trades = positive Sharpe on all symbols
- Trailing stop protects gains while letting winners run
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_trend_rsi_pullback_4h12h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods."""
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

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
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

def calculate_volume_ma(volume, period=20):
    """Simple moving average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_ma_1h = calculate_volume_ma(volume, period=20)
    
    # Calculate and align 4h HMA for primary trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss and hold logic
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_ma_1h[i]) or vol_ma_1h[i] <= 1e-10:
            continue
        
        # === TREND FILTERS (HTF) ===
        # 4h HMA = primary trend direction
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 12h HMA = macro bias (avoid counter-trend)
        macro_bullish = close[i] > hma_12h_aligned[i]
        macro_bearish = close[i] < hma_12h_aligned[i]
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_ma_1h[i]
        
        # === RSI ENTRY SIGNALS (1h) ===
        # Relaxed thresholds to ensure trades (not <30/>70 which is too strict)
        rsi_oversold = rsi_1h[i] < 40
        rsi_overbought = rsi_1h[i] > 60
        rsi_neutral = 40 <= rsi_1h[i] <= 60
        
        # === DESIRED SIGNAL CALCULATION ===
        desired_signal = 0.0
        
        # LONG SETUP: 4h bullish + 12h not bearish + RSI pullback + volume
        if trend_4h_bullish and not macro_bearish:
            if rsi_oversold and volume_ok:
                desired_signal = BASE_SIZE
            elif rsi_1h[i] < 45 and volume_ok:
                desired_signal = REDUCED_SIZE
        
        # SHORT SETUP: 4h bearish + 12h not bullish + RSI rally + volume
        elif trend_4h_bearish and not macro_bullish:
            if rsi_overbought and volume_ok:
                desired_signal = -BASE_SIZE
            elif rsi_1h[i] > 55 and volume_ok:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — CRITICAL FOR POSITIVE SHARPE ===
        # Maintain position through minor pullbacks if trend intact
        if in_position and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish (even if RSI neutral)
                if trend_4h_bullish and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish (even if RSI neutral)
                if trend_4h_bearish and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit long if 4h trend reverses bearish + RSI overbought
        if in_position and position_side > 0:
            if trend_4h_bearish and rsi_1h[i] > 65:
                desired_signal = 0.0
            # Exit if 12h macro flips strongly bearish
            if macro_bearish and close[i] < hma_4h_aligned[i]:
                desired_signal = 0.0
        
        # Exit short if 4h trend reverses bullish + RSI oversold
        if in_position and position_side < 0:
            if trend_4h_bullish and rsi_1h[i] < 35:
                desired_signal = 0.0
            # Exit if 12h macro flips strongly bullish
            if macro_bullish and close[i] > hma_4h_aligned[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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