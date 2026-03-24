#!/usr/bin/env python3
"""
Experiment #553: 5m Primary + 15m/4h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 5m timeframe is unexplored territory. By using 15m/4h HMA for trend bias
and 5m RSI pullback for entry timing, we can capture intraday moves with HTF confirmation.
Session filter (08-20 UTC) avoids low-volume Asian session whipsaws. This combines:
- 15m HMA(21) for short-term trend bias
- 4h HMA(21) for medium-term trend bias  
- 5m RSI(14) pullback to EMA(21) for entry timing
- Session filter: only trade 08-20 UTC (London/NY active hours)
- ATR(14)*2.5 stoploss on all positions
- Size = 0.15-0.20 (smaller due to higher trade frequency)

Key differences from failed 15m strategies (#541, #545, #549):
1. 5m has more entry opportunities than 15m (finer granularity)
2. Dual HTF confirmation (15m + 4h) vs single HTF
3. Session filter mandatory for 5m to avoid Asian session noise
4. Looser RSI entry (30-70 range) to ensure sufficient trades
5. EMA21 pullback entry (price retraces to EMA then continues trend)

Strategy logic:
1. 4h HMA(21) = medium trend bias (slow filter)
2. 15m HMA(21) = short trend bias (faster filter)
3. 5m EMA(21) = entry pullback level
4. 5m RSI(14) = entry timing (oversold in uptrend, overbought in downtrend)
5. Session filter: open_time hour 08-20 UTC only
6. ATR(14)*2.5 stoploss on all positions

Entry conditions (must ALL be true):
- LONG: 4h HMA bull + 15m HMA bull + 5m price > 5m EMA + RSI 30-50 + session active
- SHORT: 4h HMA bear + 15m HMA bear + 5m price < 5m EMA + RSI 50-70 + session active

Target: Sharpe>0.40, trades>=100 train (25/year), trades>=10 test
Timeframe: 5m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_hma_rsi_pullback_15m4h_session_v1"
timeframe = "5m"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def is_session_active(open_time_unix):
    """
    Check if timestamp is within active trading session (08-20 UTC)
    open_time_unix: Unix timestamp in milliseconds
    """
    # Convert to hours UTC
    hour_utc = (open_time_unix // 1000 // 3600) % 24
    return 8 <= hour_utc <= 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 15m HMA for short trend bias
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    # Calculate and align 4h HMA for medium trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 5m indicators
    ema_5m = calculate_ema(close, period=21)
    rsi_5m = calculate_rsi(close, period=14)
    atr_5m = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_5m[i]) or atr_5m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_5m[i]) or np.isnan(rsi_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        session_active = is_session_active(open_time[i])
        
        if not session_active:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h medium + 15m short) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_15m_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_15m_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === 5M TREND (price vs EMA21) ===
        price_above_ema = close[i] > ema_5m[i]
        price_below_ema = close[i] < ema_5m[i]
        
        # EMA slope (5-bar lookback)
        ema_slope_bull = ema_5m[i] > ema_5m[i-5] if i >= 5 and not np.isnan(ema_5m[i-5]) else False
        ema_slope_bear = ema_5m[i] < ema_5m[i-5] if i >= 5 and not np.isnan(ema_5m[i-5]) else False
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_5m[i] < 45.0
        rsi_overbought = rsi_5m[i] > 55.0
        rsi_neutral = 40.0 <= rsi_5m[i] <= 60.0
        
        # RSI recovering from oversold
        rsi_rising = rsi_5m[i] > rsi_5m[i-1] if i > 0 else False
        rsi_falling = rsi_5m[i] < rsi_5m[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: HTF bull + price above EMA + RSI pullback (30-50) + RSI rising
        if htf_bull and price_above_ema and ema_slope_bull:
            if rsi_oversold and rsi_rising:
                desired_signal = SIZE_STRONG
            elif rsi_neutral and rsi_rising:
                desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + price below EMA + RSI pullback (50-70) + RSI falling
        elif htf_bear and price_below_ema and ema_slope_bear:
            if rsi_overbought and rsi_falling:
                desired_signal = -SIZE_STRONG
            elif rsi_neutral and rsi_falling:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_5m[i]
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