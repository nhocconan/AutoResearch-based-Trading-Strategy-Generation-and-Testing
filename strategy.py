#!/usr/bin/env python3
"""
Experiment #705: 15m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Vol Filter

Hypothesis: 15m timeframe is underexplored but dangerous due to fee drag. Key insight from
#701 failure (-80% return): 15m needs EXTREMELY selective entries with strong HTF confirmation.
This strategy uses 4h/1d HMA for trend direction (call ONCE before loop), 15m RSI(7) for
pullback entries in trend direction, ATR ratio vol filter to avoid chop, and session filter
for high-volume hours. Discrete sizing 0.15-0.25 for 15m frequency.

Key innovations:
1. 4h HMA(21) + 1d HMA(21) dual HTF confirmation - both must align for entry
2. RSI(7) extreme pullback - <25 for long in uptrend, >75 for short in downtrend
3. ATR(7)/ATR(30) ratio > 1.2 - requires vol expansion, avoids dead chop
4. Session filter 00-14 UTC - London/NY overlap for better fills
5. 2.5x ATR trailing stop - tight risk management for 15m noise
6. Discrete sizing: 0.0, ±0.15, ±0.20, ±0.25 to minimize fee churn

Entry conditions (SELECTIVE for 40-100 trades/year target):
- LONG: 4h HMA bull + 1d HMA bull + RSI(7)<25 + ATR ratio>1.2 + session
- SHORT: 4h HMA bear + 1d HMA bear + RSI(7)>75 + ATR ratio>1.2 + session

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller than 6h due to higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi7_vol_4h1d_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
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
    """Average True Range - volatility measure for stops"""
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    hma_21 = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR ratio for vol expansion filter
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    SIZE_MED = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(rsi_7[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === Session filter (00-14 UTC for London/NY overlap) ===
        try:
            hour = pd.to_datetime(prices['open_time'].iloc[i], unit='ms').hour
            in_session = 0 <= hour <= 14
        except:
            in_session = True  # Default to true if parsing fails
        
        # === HTF BIAS (4h and 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI ENTRY (Extreme pullback in trend direction) ===
        # Long on deep oversold in uptrend
        rsi_long = rsi_7[i] < 25.0
        # Short on deep overbought in downtrend
        rsi_short = rsi_7[i] > 75.0
        
        # === Volatility filter (ATR ratio expansion) ===
        vol_expand = atr_ratio[i] > 1.2
        
        # === ENTRY LOGIC (SELECTIVE - 40-100 trades/year target) ===
        desired_signal = 0.0
        
        # LONG: Full HTF alignment + RSI extreme + vol expansion + session
        if htf_4h_bull and htf_1d_bull and rsi_long and vol_expand and in_session:
            desired_signal = SIZE_STRONG
        # LONG medium: HTF alignment + RSI extreme (no vol filter)
        elif htf_4h_bull and htf_1d_bull and rsi_long and in_session:
            desired_signal = SIZE_MED
        # LONG base: 4h bias + RSI extreme + session (looser)
        elif htf_4h_bull and rsi_long and in_session:
            desired_signal = SIZE_BASE
        
        # SHORT: Full HTF alignment + RSI extreme + vol expansion + session
        elif htf_4h_bear and htf_1d_bear and rsi_short and vol_expand and in_session:
            desired_signal = -SIZE_STRONG
        # SHORT medium: HTF alignment + RSI extreme (no vol filter)
        elif htf_4h_bear and htf_1d_bear and rsi_short and in_session:
            desired_signal = -SIZE_MED
        # SHORT base: 4h bias + RSI extreme + session (looser)
        elif htf_4h_bear and rsi_short and in_session:
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
        elif desired_signal >= SIZE_MED * 0.9:
            final_signal = SIZE_MED
        elif desired_signal <= -SIZE_MED * 0.9:
            final_signal = -SIZE_MED
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