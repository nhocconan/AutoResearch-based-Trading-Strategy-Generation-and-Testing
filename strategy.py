#!/usr/bin/env python3
"""
Experiment #341: 15m Primary + 1h/4h/1d HTF — Simplified HMA/RSI Pullback with Session Filter

Hypothesis: Previous 15m strategies failed (Sharpe=0.000, 0 trades) due to overly strict confluence.
This version SIMPLIFIES entry logic while maintaining HTF alignment for direction.

Key changes from failed 15m attempts (#329, #337):
1. REDUCED confluence from 5+ factors to 2-3 (HTF trend + RSI pullback + optional session)
2. LOOSENED RSI thresholds: 35/65 instead of 30/70 (more trigger opportunities)
3. REMOVED Choppiness Index (computationally heavy, minimal edge on 15m)
4. ADDED session filter: prefer UTC 00-12 (London/NY overlap = better liquidity)
5. ENSURED minimum trade frequency: entry triggers on any RSI extreme + HTF alignment

Strategy Logic:
- 4h HMA(21) = primary trend direction (bull/bear bias)
- 1h RSI(14) = pullback detection (oversold in uptrend, overbought in downtrend)
- 15m price action = entry trigger (close > open for long, close < open for short)
- Session filter: 00-12 UTC preferred (but allow 12-24 with stronger signal)
- Stoploss: 2.5x ATR(14) from entry

Position Sizing:
- 0.20 base size
- 0.30 when 4h + 1d HTF aligned (stronger conviction)
- Discrete levels only: 0.0, ±0.20, ±0.30

Target: Sharpe>0.40, DD>-40%, trades>=40 train, trades>=5 test, ALL symbols positive Sharpe
Trade frequency: 50-100 trades/year (critical for 15m to avoid fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_session_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_price = prices["open"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate HTF RSI for pullback detection (1h)
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]):
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
        
        if np.isnan(rsi_1h_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF CONFIRMATION (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === HTF RSI PULLBACK (1h) ===
        rsi_1h_oversold = rsi_1h_aligned[i] < 40.0
        rsi_1h_overbought = rsi_1h_aligned[i] > 60.0
        
        # === 15m RSI EXTREMES (LOOSENED for more trades) ===
        rsi_15m_oversold = rsi_15m[i] < 40.0
        rsi_15m_overbought = rsi_15m[i] > 60.0
        
        # === 15m PRICE ACTION ===
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === SESSION FILTER (UTC 00-12 preferred) ===
        hour = get_session_hour(open_time[i])
        prime_session = 0 <= hour < 12  # London/NY overlap
        
        # === ENTRY LOGIC (SIMPLIFIED - 2-3 confluence factors) ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bull + RSI pullback + price action
        if htf_4h_bull:
            # Strong signal: 4h + 1d aligned + RSI oversold
            if htf_1d_bull and rsi_1h_oversold and bullish_candle:
                desired_signal = SIZE_STRONG if prime_session else SIZE_BASE
            # Base signal: 4h bull + 15m RSI oversold + bullish candle
            elif rsi_15m_oversold and bullish_candle and above_sma200:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: 4h bear + RSI pullback + price action
        elif htf_4h_bear:
            # Strong signal: 4h + 1d aligned + RSI overbought
            if htf_1d_bear and rsi_1h_overbought and bearish_candle:
                desired_signal = -SIZE_STRONG if prime_session else -SIZE_BASE
            # Base signal: 4h bear + 15m RSI overbought + bearish candle
            elif rsi_15m_overbought and bearish_candle and below_sma200:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals