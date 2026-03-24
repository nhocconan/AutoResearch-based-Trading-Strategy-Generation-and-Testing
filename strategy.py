#!/usr/bin/env python3
"""
Experiment #013: 5m Primary + 15m/4h HTF — Session-Filtered Trend Following

Hypothesis: 5m timeframe has ZERO prior experiments. Key insight from research:
- Lower TF strategies fail due to fee drag from excessive trades
- SOLUTION: Use 4h HMA for major trend bias + 15m RSI for momentum + 5m for entry timing
- Session filter (08-20 UTC) avoids low-liquidity whipsaws (Asian overnight = choppy)
- Volume confirmation ensures only trade on real moves, not noise
- This combines HTF trend direction with LTF entry precision = fewer trades, higher quality
- Position size 0.18 (smaller due to more trades = fee sensitivity)
- Target: 50-120 trades/year, Sharpe>0.019 (beat current best)

Key design choices:
- Timeframe: 5m (requires extreme selectivity)
- HTF: 4h HMA(21) for major trend, 15m RSI(14) for momentum confirmation
- Session filter: 08-20 UTC only (high liquidity hours)
- Entry: 5m EMA crossover + volume spike + HTF alignment
- Stoploss: 2.0x ATR trailing (tighter for 5m noise)
- Size: 0.18 (18% of capital, conservative for 5m fee drag)

Target: Sharpe>0.019, DD>-50%, trades>=10 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_hma_rsi_volume_4h15m_v1"
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs recent average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / (vol_avg + 1e-10)
    return ratio

def is_session_active(open_time):
    """
    Session filter: 08-20 UTC only (high liquidity hours)
    open_time is in milliseconds since epoch
    """
    # Convert to hour of day UTC
    hour = (open_time // (1000 * 60 * 60)) % 24
    return 8 <= hour < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_15m = get_htf_data(prices, '15m')
    
    # Calculate and align 4h HMA for major trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m RSI for momentum confirmation
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate primary (5m) indicators
    ema_fast = calculate_ema(close, period=8)
    ema_slow = calculate_ema(close, period=21)
    rsi_5m = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (conservative for 5m fee drag)
    
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
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_5m[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        session_active = is_session_active(open_time[i])
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF MOMENTUM (15m RSI) ===
        # RSI > 50 = bullish momentum, RSI < 50 = bearish momentum
        htf_momentum_bull = rsi_15m_aligned[i] > 50.0
        htf_momentum_bear = rsi_15m_aligned[i] < 50.0
        
        # === 5m EMA CROSSOVER ===
        ema_cross_bull = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_bear = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # === 5m EMA TREND (sustained) ===
        ema_trend_bull = ema_fast[i] > ema_slow[i]
        ema_trend_bear = ema_fast[i] < ema_slow[i]
        
        # === VOLUME CONFIRMATION ===
        volume_spike = vol_ratio[i] > 1.5  # 50% above average
        
        # === RSI FILTER (5m) ===
        rsi_ok_long = rsi_5m[i] > 40.0 and rsi_5m[i] < 70.0  # not oversold/overbought
        rsi_ok_short = rsi_5m[i] > 30.0 and rsi_5m[i] < 60.0
        
        # === DESIRED SIGNAL (Multi-confluence) ===
        desired_signal = 0.0
        
        # LONG: session active + HTF bull + HTF momentum bull + EMA trend + volume
        if session_active and htf_bull and htf_momentum_bull and ema_trend_bull and volume_spike and rsi_ok_long:
            desired_signal = SIZE
        # Entry on crossover with all filters
        elif session_active and ema_cross_bull and htf_bull and htf_momentum_bull and rsi_5m[i] > 45.0:
            desired_signal = SIZE
        
        # SHORT: session active + HTF bear + HTF momentum bear + EMA trend + volume
        elif session_active and htf_bear and htf_momentum_bear and ema_trend_bear and volume_spike and rsi_ok_short:
            desired_signal = -SIZE
        # Entry on crossover with all filters
        elif session_active and ema_cross_bear and htf_bear and htf_momentum_bear and rsi_5m[i] < 55.0:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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
                # Flip position
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