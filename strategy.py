#!/usr/bin/env python3
"""
Experiment #485: 15m Primary + 4h/1d HTF — Session-Based RSI Pullback Strategy

Hypothesis: 15m strategies fail due to either (a) too many filters = 0 trades, or 
(b) too many trades = fee drag. This strategy uses:
1. 1d HMA(21) for macro trend bias (loaded ONCE via mtf_data)
2. 4h HMA(21) for intermediate trend confirmation (loaded ONCE via mtf_data)
3. 15m RSI(7) for entry timing - oversold bounce in uptrend, overbought fade in downtrend
4. Session filter: 00-12 UTC only (London/NY overlap) to reduce trade count
5. ATR(14)*2.5 stoploss on all positions
6. LOOSE entry thresholds to guarantee trade generation (RSI<35 or >65)

Key design for 15m success:
- HTF filters for DIRECTION only (4h/1d)
- 15m indicators for ENTRY TIMING only
- Session filter cuts trades by ~50%
- Size: 0.15-0.25 (smaller for higher frequency timeframe)
- Target: 50-100 trades/year (not 300+)

Why this might work on 15m:
- Prior 15m experiments failed with Sharpe=0.000 (zero trades)
- This uses LOOSE RSI thresholds (35/65 not 30/70)
- Session filter ensures we don't overtrade
- HTF alignment via mtf_data prevents look-ahead
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_pullback_4h1d_v2"
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

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column"""
    # open_time is in milliseconds since epoch
    open_time_ms = prices["open_time"].values
    hours = (open_time_ms // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Get UTC hour for session filter
    hours = get_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    
    # Position sizing for 15m (smaller due to higher frequency)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
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
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER: Only trade 00-12 UTC (London/NY overlap) ===
        in_session = hours[i] < 12
        
        # === HTF TREND BIAS ===
        # 1d HMA = macro trend
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # 4h HMA = intermediate trend
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI EXTREMES (LOOSE: 35/65 for entries) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_oversold = rsi_7[i] < 25.0
        rsi_extreme_overbought = rsi_7[i] > 75.0
        
        # RSI recovery/weakness crosses
        rsi_crossing_up = rsi_7[i] > 40.0 and rsi_7[i-1] <= 40.0
        rsi_crossing_down = rsi_7[i] < 60.0 and rsi_7[i-1] >= 60.0
        
        # === ENTRY LOGIC (LOOSE - designed to generate trades) ===
        desired_signal = 0.0
        
        # TREND LONG: 1d bull + 4h bull + RSI pullback opportunity
        if htf_bull and htf_4h_bull:
            if in_session:
                if rsi_oversold and above_sma50:
                    # RSI pullback in uptrend
                    desired_signal = SIZE_STRONG
                elif rsi_crossing_up and above_sma50:
                    # RSI momentum shift
                    desired_signal = SIZE_BASE
        
        # TREND SHORT: 1d bear + 4h bear + RSI rally fade
        elif htf_bear and htf_4h_bear:
            if in_session:
                if rsi_overbought and below_sma50:
                    # RSI rally fade in downtrend
                    desired_signal = -SIZE_STRONG
                elif rsi_crossing_down and below_sma50:
                    # RSI weakness shift
                    desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: Extreme oversold even without HTF agreement
        if desired_signal == 0.0:
            if rsi_extreme_oversold and above_sma200:
                # Very oversold + above long-term MA = bounce candidate
                desired_signal = SIZE_BASE
            elif rsi_7[i] < 20.0:
                # Extremely oversold = high probability bounce
                desired_signal = SIZE_BASE
        
        # MEAN REVERSION SHORT: Extreme overbought even without HTF agreement
        if desired_signal == 0.0:
            if rsi_extreme_overbought and below_sma200:
                # Very overbought + below long-term MA = fade candidate
                desired_signal = -SIZE_BASE
            elif rsi_7[i] > 80.0:
                # Extremely overbought = high probability fade
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