#!/usr/bin/env python3
"""
Experiment #557: 15m Primary + 4h/12h HTF — RSI Pullback with Session Filter

Hypothesis: 15m timeframe has ZERO successful experiments because entry conditions
were TOO STRICT (RSI<25, multiple conflicting filters). This strategy LOOSENS entries:
- RSI(7)<35 for long (not <25), RSI(7)>65 for short (not >75)
- Only 4h HMA for trend bias (not 4h+1d+1w which rarely align)
- Session filter 00-12 UTC to capture London/NY overlap volume
- ATR(14)*2.0 stoploss for risk management
- Target: 50-100 trades/year (not >300 which kills with fees)

Key differences from failed #545/#549/#553 (all Sharpe=0.000 = 0 trades):
1. RSI threshold loosened: 35/65 instead of 25/75
2. Single HTF filter (4h HMA) instead of triple (4h+1d+1w)
3. Session filter reduces trades to target range (not too many, not zero)
4. Simpler logic = more confluence opportunities
5. Discrete signal sizes: 0.0, ±0.15, ±0.25 to minimize fee churn

Strategy logic:
1. 4h HMA(21) = trend direction bias (aligned properly with shift(1))
2. 15m RSI(7) = entry trigger (faster than RSI(14))
3. 15m SMA(50) = momentum confirmation (price above/below)
4. Session filter: 00-12 UTC only (high volume periods)
5. ATR(14)*2.0 stoploss on all positions
6. Size: 0.25 base, 0.15 reduced for weaker signals

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_hma_4h_session_v4"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
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

def calculate_sma(close, period):
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
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)  # Standard RSI for confirmation
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15
    
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
        
        if np.isnan(rsi[i]) or np.isnan(sma_50[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC) ===
        # Convert open_time (milliseconds) to hour
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        in_session = 0 <= hour_utc <= 12  # London + NY overlap
        
        # === 4H TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15M MOMENTUM ===
        mom_bull = close[i] > sma_50[i]
        mom_bear = close[i] < sma_50[i]
        
        # === RSI EXTREMES (LOOSENED for more trades) ===
        rsi_oversold = rsi[i] < 35.0  # Was <25 in failed attempts
        rsi_overbought = rsi[i] > 65.0  # Was >75 in failed attempts
        rsi_neutral = 40.0 <= rsi[i] <= 60.0
        
        # === ENTRY LOGIC (SIMPLIFIED for more confluence) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + RSI oversold + in session + momentum confirmation
        if htf_bull and rsi_oversold and in_session:
            if mom_bull:
                desired_signal = SIZE_BASE  # Strong confluence
            else:
                desired_signal = SIZE_REDUCED  # Weaker signal
        
        # SHORT: 4h bear + RSI overbought + in session + momentum confirmation
        elif htf_bear and rsi_overbought and in_session:
            if mom_bear:
                desired_signal = -SIZE_BASE  # Strong confluence
            else:
                desired_signal = -SIZE_REDUCED  # Weaker signal
        
        # Additional entry: RSI recovery from extreme (more trades)
        if i > 1 and not np.isnan(rsi[i-1]):
            # Long: RSI was very oversold, now rising
            if rsi[i-1] < 30.0 and rsi[i] > rsi[i-1] and htf_bull and in_session:
                desired_signal = max(desired_signal, SIZE_REDUCED)
            # Short: RSI was very overbought, now falling
            elif rsi[i-1] > 70.0 and rsi[i] < rsi[i-1] and htf_bear and in_session:
                desired_signal = min(desired_signal, -SIZE_REDUCED)
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
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
        if desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_REDUCED * 0.9:
            final_signal = SIZE_REDUCED
        elif desired_signal <= -SIZE_REDUCED * 0.9:
            final_signal = -SIZE_REDUCED
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