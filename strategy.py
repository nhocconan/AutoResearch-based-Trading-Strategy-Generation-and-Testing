#!/usr/bin/env python3
"""
Experiment #090: 1h Primary + 4h/12h HTF — Mean Reversion with HTF Trend Filter

Hypothesis: After 75+ failed experiments, trend-following on BTC/ETH consistently fails
(Sharpe negative or zero). The 2022 crash and 2025 bear market destroy trend strategies.

NEW APPROACH: Mean reversion WITHIN HTF trend direction.
- 12h HMA = major trend bias (ONLY trade longs when price > 12h HMA)
- 1h Bollinger Bands = mean reversion entry (buy at lower band, sell at upper)
- 4h RSI = momentum confirmation (avoid entering against momentum)
- Session filter = only 8-20 UTC (highest volume, avoid Asian session whipsaw)
- Very strict confluence: ALL 4 conditions must align

Why this might work:
1. Mean reversion works in bear/range markets (2022, 2025)
2. HTF filter prevents counter-trend trades
3. Session filter reduces noise and trade count
4. Bollinger squeeze detection avoids low-volatility traps

Target: Sharpe>0.351, DD>-40%, trades=30-80/year on train, trades>=3 on test
Position size: 0.20 (smaller for 1h to minimize fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_bb_meanrev_htf_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = close_series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    close_series = pd.Series(close)
    sma = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bb_width(upper, lower, sma):
    """Bollinger Band Width - detects squeeze (low vol)"""
    width = (upper - lower) / sma
    return width

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h RSI for momentum confirmation
    rsi_4h_raw = calculate_rsi(df_4h['close'].values, period=14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_raw)
    
    # Calculate primary (1h) indicators
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    rsi_1h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    
    # Calculate BB width percentile for squeeze detection (lookback 100 bars)
    bb_width_pct = np.full(n, np.nan)
    for i in range(100, n):
        if not np.isnan(bb_width[i]):
            lookback = bb_width[i-100:i+1]
            lookback = lookback[~np.isnan(lookback)]
            if len(lookback) > 10:
                bb_width_pct[i] = np.sum(lookback[:-1] < bb_width[i]) / len(lookback[:-1])
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (smaller for 1h to reduce fee drag)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(rsi_1h[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(rsi_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === HTF TREND BIAS (12h HMA) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === 4h MOMENTUM CONFIRMATION (RSI) ===
        # For longs: 4h RSI should not be overbought (>70 blocks long)
        # For shorts: 4h RSI should not be oversold (<30 blocks short)
        mom_ok_long = rsi_4h_aligned[i] < 70.0
        mom_ok_short = rsi_4h_aligned[i] > 30.0
        
        # === BB SQUEEZE FILTER (avoid low vol) ===
        # Only trade when BB width is NOT at extreme low (avoid squeeze breakouts)
        no_squeeze = bb_width_pct[i] > 0.20  # width above 20th percentile
        
        # === MEAN REVERSION ENTRY (1h Bollinger Bands) ===
        # Long: price touches/pierces lower band + all filters pass
        # Short: price touches/pierces upper band + all filters pass
        price_at_lower = close[i] <= bb_lower[i] * 1.002  # within 0.2% of lower band
        price_at_upper = close[i] >= bb_upper[i] * 0.998  # within 0.2% of upper band
        
        # === DESIRED SIGNAL (ALL CONDITIONS MUST ALIGN) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + mom ok + no squeeze + price at lower band + in session
        if htf_bull and mom_ok_long and no_squeeze and price_at_lower and in_session:
            desired_signal = SIZE
        
        # SHORT: HTF bear + mom ok + no squeeze + price at upper band + in session
        elif htf_bear and mom_ok_short and no_squeeze and price_at_upper and in_session:
            desired_signal = -SIZE
        
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
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals