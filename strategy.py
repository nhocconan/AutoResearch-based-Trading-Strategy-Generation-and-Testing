#!/usr/bin/env python3
"""
Experiment #721: 15m Primary + 1h/4h HTF — Simplified Trend Pullback Strategy

Hypothesis: 15m strategies fail due to OVERLY STRICT entry conditions (0 trades).
This strategy uses SIMPLE, LOOSE conditions to ENSURE trade generation:
1. 4h HMA(21) for trend direction (HTF bias)
2. 1h RSI(14) for momentum confirmation (loose: 35/65 thresholds)
3. 15m price pullback to EMA(21) for entry timing
4. Session filter: 00-12 UTC (London/NY overlap)
5. ATR(14) 2.5x trailing stop

Key innovations for 15m success:
- LOOSE RSI thresholds (35/65 not 20/80) to ensure trades
- Only 2 confluence required (HTF trend + 1h momentum) not 3+
- Session filter reduces noise during low-volume hours
- Discrete sizing: 0.0, ±0.20, ±0.25, ±0.30
- Target: 50-100 trades/year (not 300+)

Why this should work where #709, #717 failed:
- Those had 0 trades = conditions too strict
- This uses simpler logic with lower thresholds
- HTF bias (4h) + momentum (1h) + timing (15m) = proven combo

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 15m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_session_4h1h_v1"
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

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    # Convert to hours UTC
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    hma_21 = calculate_hma(close, period=21)
    ema_21 = calculate_ema(close, period=21)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_MED = 0.25
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(hma_21[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London/NY overlap) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = (hour >= 0 and hour < 12)
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === MOMENTUM (1h RSI) - LOOSE THRESHOLDS ===
        # Use 35/65 instead of 20/80 to ensure more trades
        mom_bull = rsi_1h_aligned[i] > 45.0  # Loose bullish momentum
        mom_bear = rsi_1h_aligned[i] < 55.0  # Loose bearish momentum
        
        # === 15m PULLBACK ENTRY ===
        # Long: price pulls back to EMA21 in uptrend
        # Short: price rallies to EMA21 in downtrend
        pullback_long = (close[i] <= ema_21[i] * 1.002) and (close[i] >= ema_21[i] * 0.998)
        pullback_short = (close[i] >= ema_21[i] * 0.998) and (close[i] <= ema_21[i] * 1.002)
        
        # === 15m RSI CONFIRMATION - LOOSE ===
        rsi_oversold = rsi_14[i] < 45.0  # Was 30, now 45 for more trades
        rsi_overbought = rsi_14[i] > 55.0  # Was 70, now 55 for more trades
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 1h momentum + (pullback OR RSI oversold)
        # Only 2 of 3 conditions needed (not all 3)
        if htf_4h_bull and mom_bull:
            if pullback_long or rsi_oversold:
                if in_session:
                    desired_signal = SIZE_MED
                else:
                    desired_signal = SIZE_BASE  # Smaller size outside session
        
        # LONG: Very strong 4h bull + any momentum
        elif htf_4h_bull and close[i] > hma_4h_aligned[i] * 1.01:
            if rsi_14[i] < 50.0:  # Any dip
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 1h momentum + (pullback OR RSI overbought)
        elif htf_4h_bear and mom_bear:
            if pullback_short or rsi_overbought:
                if in_session:
                    desired_signal = -SIZE_MED
                else:
                    desired_signal = -SIZE_BASE  # Smaller size outside session
        
        # SHORT: Very strong 4h bear + any momentum
        elif htf_4h_bear and close[i] < hma_4h_aligned[i] * 0.99:
            if rsi_14[i] > 50.0:  # Any rally
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
                entry_atr = atr[i]
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