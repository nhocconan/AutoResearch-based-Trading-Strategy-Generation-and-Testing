#!/usr/bin/env python3
"""
Experiment #641: 15m Primary + 1h/4h/1d HTF — Multi-TF Trend Pullback with Session Filter

Hypothesis: 15m timeframe has ZERO successful experiments because entry conditions were TOO STRICT.
This strategy uses LOOSE entry conditions to ensure trades generate, while using HTF for direction bias.

Key innovations:
1. 4h HMA(21) for primary trend bias (proven in best strategies)
2. 1h RSI(14) for momentum confirmation (not extreme - just >45/<55)
3. 15m EMA(21) pullback entry in trend direction
4. Session filter: 00-12 UTC preferred (London/NY overlap) but not blocking
5. ATR(14) trailing stop at 2.5x for risk management
6. Discrete sizing: 0.20 base, 0.25 strong signal

Why this should work on 15m:
- Previous 15m strategies had Sharpe=0.000 (ZERO trades) due to over-filtering
- This uses minimal filters: just HTF trend + basic momentum + pullback
- Session filter reduces size but doesn't block entries
- Target: 50-80 trades/year (within 40-100 target for 15m)

CRITICAL: Entry conditions LOOSE to ensure >=10 trades per symbol on train.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_pullback_rsi_1h4h_session_v1"
timeframe = "15m"
leverage = 1.0

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
    
    rs = np.zeros(n)
    rs[:] = np.nan
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
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

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    hour = (ts_seconds % 86400) / 3600.0
    return int(hour)

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
    ema_15m = calculate_ema(close, period=21)
    rsi_15m = calculate_rsi(close, period=14)
    atr_15m = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_15m[i]) or np.isnan(rsi_15m[i]):
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
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h MOMENTUM (RSI) ===
        # Loose filter: just avoid extreme overbought/oversold against trend
        momentum_long = rsi_1h_aligned[i] > 40.0  # Not deeply oversold
        momentum_short = rsi_1h_aligned[i] < 60.0  # Not deeply overbought
        
        # === 15m PULLBACK ENTRY ===
        # Long: price pulls back to EMA in uptrend
        pullback_long = close[i] <= ema_15m[i] * 1.002 and close[i] >= ema_15m[i] * 0.995
        # Short: price rallies to EMA in downtrend
        pullback_short = close[i] >= ema_15m[i] * 0.998 and close[i] <= ema_15m[i] * 1.005
        
        # === BREAKOUT ENTRY (alternative) ===
        breakout_long = close[i] > ema_15m[i] * 1.005 and rsi_15m[i] > 50
        breakout_short = close[i] < ema_15m[i] * 0.995 and rsi_15m[i] < 50
        
        # === SESSION FILTER ===
        hour = get_session_hour(open_time[i])
        prime_session = 0 <= hour <= 12  # London/NY overlap preferred
        session_multiplier = 1.0 if prime_session else 0.7  # Reduce size outside prime hours
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + momentum ok + (pullback OR breakout)
        if htf_bull and momentum_long:
            if pullback_long:
                desired_signal = SIZE_STRONG
            elif breakout_long:
                desired_signal = SIZE_BASE
            elif close[i] > ema_15m[i] and rsi_15m[i] > 45:
                # Very loose: just above EMA with decent RSI
                desired_signal = SIZE_BASE * 0.7
        
        # SHORT: HTF bear + momentum ok + (pullback OR breakout)
        elif htf_bear and momentum_short:
            if pullback_short:
                desired_signal = -SIZE_STRONG
            elif breakout_short:
                desired_signal = -SIZE_BASE
            elif close[i] < ema_15m[i] and rsi_15m[i] < 55:
                # Very loose: just below EMA with decent RSI
                desired_signal = -SIZE_BASE * 0.7
        
        # Apply session multiplier
        desired_signal = desired_signal * session_multiplier
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
            final_signal = np.sign(desired_signal) * SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_15m[i]
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