#!/usr/bin/env python3
"""
Experiment #1590: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + Volume

Hypothesis: After 11 failed 4h experiments and multiple 1h failures with CRSI/Choppiness,
we need a different approach for lower timeframes. The Ehlers Fisher Transform excels
at catching reversals in bear/range markets (2022 crash, 2025 bear) where RSI fails.

Key innovations:
1. Fisher Transform (period=9) on 1h - catches reversals at extreme levels (-1.5/+1.5)
2. 4h HMA(21) for medium-term trend bias (align properly with shift(1))
3. 12h HMA(21) for long-term regime filter (only trade with 12h trend)
4. Volume confirmation (>0.8x 20-bar avg) - ensures meaningful moves
5. Session filter (8-20 UTC) - avoids low-liquidity Asian session whipsaw
6. ATR(14) 2.5x trailing stop for drawdown control
7. Discrete position sizing (0.25) - smaller for 1h to reduce fee impact

Why this should beat Sharpe 0.618:
- Fisher Transform outperforms RSI in bear markets (research-backed)
- Dual HTF filter (4h + 12h) ensures we only trade with major trend
- Volume + session filters reduce false signals = fewer trades, higher quality
- 1h entry timing within 4h/12h trend = HTF frequency with LTF precision
- Target: 40-70 trades/year (within 30-80 target for 1h)

Timeframe: 1h (required for this experiment)
HTF: 4h HMA + 12h HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 30/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_4h12h_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(close, period=9):
    """
    Ehlers Fisher Transform
    Converts price into a Gaussian normal distribution
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    X = 0.66 * ((close - min_low) / (max_high - min_low) - 0.5) + 0.67 * prev_X
    
    Signal: Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Calculate highest high and lowest low over period
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        hh[i] = np.max(close[i - period + 1:i + 1])
        ll[i] = np.min(close[i - period + 1:i + 1])
    
    # Calculate X value
    x = np.full(n, np.nan)
    x_raw = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        if hh[i] - ll[i] > 1e-10:
            x_raw[i] = (close[i] - ll[i]) / (hh[i] - ll[i]) - 0.5
        else:
            x_raw[i] = 0.0
        
        if i == period - 1:
            x[i] = 0.66 * x_raw[i]
        else:
            x[i] = 0.66 * x_raw[i] + 0.67 * x[i - 1] if not np.isnan(x[i - 1]) else 0.66 * x_raw[i]
    
    # Clamp X to (-0.999, 0.999) to avoid ln(0)
    x = np.clip(x, -0.999, 0.999)
    
    # Calculate Fisher Transform
    for i in range(period - 1, n):
        if abs(x[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1 + x[i]) / (1 - x[i]))
            if i > period - 1:
                fisher_prev[i] = fisher[i - 1]
    
    return fisher, fisher_prev

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
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

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000
    utc_hour = pd.to_datetime(ts_seconds, unit='s').hour
    return utc_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for long-term regime
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(close, period=9)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h to reduce fee impact
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        session_active = 8 <= utc_hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] >= 0.8 * vol_sma[i]
        
        # === TREND BIAS (4h HMA + 12h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Fisher long + 4h bull + 12h bull (or neutral) + session + volume
        if fisher_long and hma_4h_bull and (hma_12h_bull or not hma_12h_bear) and session_active and volume_confirmed:
            desired_signal = BASE_SIZE
        
        # SHORT: Fisher short + 4h bear + 12h bear (or neutral) + session + volume
        elif fisher_short and hma_4h_bear and (hma_12h_bear or not hma_12h_bull) and session_active and volume_confirmed:
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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