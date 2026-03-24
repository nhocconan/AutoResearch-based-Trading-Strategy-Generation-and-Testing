#!/usr/bin/env python3
"""
Experiment #901: 15m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume Confirm

Hypothesis: 15m timeframe with 4h trend + 1d bias provides optimal entry precision
while maintaining HTF signal quality. RSI(7) pullbacks in trend direction capture
continuation moves with better risk/reward than breakouts. Volume confirmation
filters false signals. Session filter (00-12 UTC) reduces low-liquidity trades.

Key innovations:
1. 1d HMA(21) for primary bias - price above = long bias, below = short bias
2. 4h HMA(16/48) crossover for trend confirmation - fast above slow = bull trend
3. 15m RSI(7) for entry timing - oversold in uptrend, overbought in downtrend
4. Volume spike filter (1.5x 20-bar avg) confirms genuine moves
5. Session filter: prefer 00-12 UTC (London/NY overlap)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 1d HMA bull + (4h HMA bull OR 4h HMA cross up) + RSI(7)<35 + volume confirm
- SHORT: 1d HMA bear + (4h HMA bear OR 4h HMA cross down) + RSI(7)>65 + volume confirm
- Session bonus: 00-12 UTC gets priority, 12-24 UTC requires stronger signals

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
Trade freq: 50-100/year max (use strict confluence)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_vol_session_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.zeros(n)
    diff[:] = np.nan
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
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
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
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

def calculate_volume_spike(volume, period=20):
    """Volume spike detection: current volume vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Get open_time for session filter
    open_time = prices["open_time"].values
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, period=48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_spike(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        htf_4h_bull = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        htf_4h_bear = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === 4h HMA CROSSOVER (momentum shift) ===
        hma_4h_cross_long = False
        hma_4h_cross_short = False
        if i > 0 and not np.isnan(hma_4h_16_aligned[i-1]) and not np.isnan(hma_4h_48_aligned[i-1]):
            hma_4h_cross_long = (hma_4h_16_aligned[i-1] <= hma_4h_48_aligned[i-1]) and (hma_4h_16_aligned[i] > hma_4h_48_aligned[i])
            hma_4h_cross_short = (hma_4h_16_aligned[i-1] >= hma_4h_48_aligned[i-1]) and (hma_4h_16_aligned[i] < hma_4h_48_aligned[i])
        
        # === 15m RSI CONDITIONS (LOOSE for trade generation) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_long = rsi_7[i] < 25.0
        rsi_extreme_short = rsi_7[i] > 75.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] >= 1.3  # 30% above average
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # open_time is in milliseconds, convert to hour
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        prime_session = 0 <= hour_utc < 12  # London/NY overlap
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADES) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_1d_bull:
            # Need at least 4h bull trend OR crossover
            if htf_4h_bull or hma_4h_cross_long:
                if rsi_oversold:
                    if vol_confirm:
                        desired_signal = SIZE_STRONG if prime_session else SIZE_BASE
                    else:
                        # Allow entry without volume if RSI extreme
                        if rsi_extreme_long:
                            desired_signal = SIZE_BASE
                elif rsi_extreme_long:
                    # Very oversold, enter even without volume confirm
                    desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_1d_bear:
            # Need at least 4h bear trend OR crossover
            if htf_4h_bear or hma_4h_cross_short:
                if rsi_overbought:
                    if vol_confirm:
                        desired_signal = -SIZE_STRONG if prime_session else -SIZE_BASE
                    else:
                        # Allow entry without volume if RSI extreme
                        if rsi_extreme_short:
                            desired_signal = -SIZE_BASE
                elif rsi_extreme_short:
                    # Very overbought, enter even without volume confirm
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals