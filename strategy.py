#!/usr/bin/env python3
"""
Experiment #101: 15m Primary + 1h/4h HTF — Pullback in Trend with Volume

Hypothesis: After 93 failed experiments, the pattern for 15m is clear:
- Too many filters = 0 trades (see #089, #093, #096, #097, #099 all Sharpe=0.000)
- SOLUTION: Simplify entry conditions, use HTF for bias only, 15m for timing
- 4h HMA(21) provides stable trend bias (changes slowly)
- 1h RSI(14) confirms momentum without being too restrictive
- 15m pullback to HMA(21) + volume spike = entry trigger
- LOOSE filters to ensure >=30 trades on train, >=3 on test
- Session filter: 00-12 UTC (London+NY overlap) for quality trades

Key design choices:
- Timeframe: 15m (target 40-100 trades/year)
- HTF: 4h HMA for trend, 1h RSI for momentum (load ONCE before loop)
- Entry: 15m pullback to HMA + volume 1.5x avg + HTF bias alignment
- Position size: 0.20 (20% of capital, conservative for 15m frequency)
- Stoploss: 2.5x ATR trailing (signal→0 when hit)
- Discrete signals: 0.0, ±0.20 to minimize fee churn

Target: Sharpe>0.167, DD>-40%, trades>=30 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pullback_hma_vol_4h1h_session_v1"
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align 4h HMA for major trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1h RSI for momentum
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_15m = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m)
    
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
        if np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
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
        
        # === SESSION FILTER (00-12 UTC for London+NY overlap) ===
        # 15m bars: 96 per day, indices 0-95 for first day
        # Assuming data starts at 00:00, bar index % 96 gives bar of day
        bar_of_day = i % 96
        in_session = bar_of_day < 48  # First 48 bars = 00:00-12:00 UTC
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h MOMENTUM (RSI) ===
        # Not extreme: 30-70 range allows entries
        mom_ok_long = rsi_1h_aligned[i] > 35.0 and rsi_1h_aligned[i] < 80.0
        mom_ok_short = rsi_1h_aligned[i] > 20.0 and rsi_1h_aligned[i] < 65.0
        
        # === 15m PULLBACK ENTRY ===
        # Long: price near/pulling back to HMA in uptrend
        pullback_long = close[i] <= hma_15m[i] * 1.005  # within 0.5% of HMA
        pullback_short = close[i] >= hma_15m[i] * 0.995  # within 0.5% of HMA
        
        # Price actually bounced off HMA (previous bar was below/above)
        bounce_long = i > 0 and close[i-1] < hma_15m[i-1] and close[i] >= hma_15m[i]
        bounce_short = i > 0 and close[i-1] > hma_15m[i-1] and close[i] <= hma_15m[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ok = vol_ratio[i] >= 1.3  # 30% above average (loose filter)
        
        # === RSI FILTER (15m) ===
        rsi_ok_long = rsi_15m[i] > 30.0 and rsi_15m[i] < 75.0
        rsi_ok_short = rsi_15m[i] > 25.0 and rsi_15m[i] < 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: HTF bull + 1h mom ok + 15m pullback/bounce + volume + session
        if htf_bull and mom_ok_long and rsi_ok_long:
            if (pullback_long or bounce_long) and vol_ok:
                if in_session:
                    desired_signal = SIZE
                else:
                    desired_signal = SIZE * 0.7  # reduced size outside session
        
        # SHORT: HTF bear + 1h mom ok + 15m pullback/bounce + volume + session
        elif htf_bear and mom_ok_short and rsi_ok_short:
            if (pullback_short or bounce_short) and vol_ok:
                if in_session:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.7  # reduced size outside session
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
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