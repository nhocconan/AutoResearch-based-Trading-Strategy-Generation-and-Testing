#!/usr/bin/env python3
"""
Experiment #693: 5m Primary + 15m/4h HTF — HMA Trend + RSI Momentum + Session Filter

Hypothesis: 5m timeframe requires extreme selectivity to avoid fee drag. Using 4h HMA for 
primary trend direction (proven edge), 15m RSI for momentum confirmation, and 5m entries 
ONLY during high-liquidity sessions (08-20 UTC). This combines HTF signal quality with 
lower TF execution precision while minimizing trades during low-liquidity periods.

Key innovations:
1. 4h HMA(21) slope filter - only trade when 4h trend is established (slope > threshold)
2. 15m RSI(14) momentum - confirms direction before 5m entry
3. Session filter (08-20 UTC) - avoids Asian session whipsaw, trades London/NY overlap
4. 5m RSI(7) fast entry - catches pullbacks within HTF trend
5. ATR(14) trailing stop - 2.0x for tight risk on lower TF
6. Discrete sizing: 0.15 base, 0.25 strong (smaller due to higher trade frequency)

Entry conditions:
- LONG: 4h HMA slope > 0 + price > 4h HMA + 15m RSI > 50 + 5m RSI < 55 (pullback) + session
- SHORT: 4h HMA slope < 0 + price < 4h HMA + 15m RSI < 50 + 5m RSI > 45 (rally) + session

Target: Sharpe>0.40, trades>=50/year, DD>-40%
Timeframe: 5m
Size: 0.15-0.25 discrete (smaller due to fee drag on 5m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_hma_rsi_session_15m4h_v1"
timeframe = "5m"
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

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope as rate of change over lookback periods"""
    n = len(hma)
    slope = np.full(n, np.nan)
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i - lookback]):
            if hma[i - lookback] > 1e-10:
                slope[i] = (hma[i] - hma[i - lookback]) / hma[i - lookback] * 100
            else:
                slope[i] = 0.0
    return slope

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
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds, convert to hours
    return (open_time // (1000 * 60 * 60)) % 24

def is_liquid_session(hour):
    """Check if hour is in liquid session (08-20 UTC - London/NY overlap)"""
    return 8 <= hour <= 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 4h HMA slope for trend strength
    hma_4h_slope_raw = calculate_hma_slope(hma_4h_raw, lookback=3)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope_raw)
    
    # Calculate and align 15m RSI for momentum
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    rsi_5m = calculate_rsi(close, period=7)  # Faster RSI for 5m entries
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        hour = get_session_hour(open_time[i])
        in_session = is_liquid_session(hour)
        
        # === 4H TREND FILTER ===
        htf_bull = close[i] > hma_4h_aligned[i] and hma_4h_slope_aligned[i] > 0.05
        htf_bear = close[i] < hma_4h_aligned[i] and hma_4h_slope_aligned[i] < -0.05
        
        # === 15M MOMENTUM CONFIRMATION ===
        mom_bull = rsi_15m_aligned[i] > 50.0
        mom_bear = rsi_15m_aligned[i] < 50.0
        
        # === 5M ENTRY (PULLBACK IN TREND) ===
        # Long: pullback in uptrend (RSI 35-55)
        entry_long = 35.0 <= rsi_5m[i] <= 55.0
        # Short: rally in downtrend (RSI 45-65)
        entry_short = 45.0 <= rsi_5m[i] <= 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h trend up + 15m momentum up + 5m pullback + session
        if htf_bull and mom_bull and entry_long and in_session:
            # Strong signal if all conditions met
            desired_signal = SIZE_STRONG
        elif htf_bull and mom_bull and in_session:
            # Weaker: just trend + momentum + session
            desired_signal = SIZE_BASE
        
        # SHORT: 4h trend down + 15m momentum down + 5m rally + session
        elif htf_bear and mom_bear and entry_short and in_session:
            # Strong signal if all conditions met
            desired_signal = -SIZE_STRONG
        elif htf_bear and mom_bear and in_session:
            # Weaker: just trend + momentum + session
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing - tighter for 5m) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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