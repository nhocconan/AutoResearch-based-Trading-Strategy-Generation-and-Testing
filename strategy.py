#!/usr/bin/env python3
"""
Experiment #097: 15m Primary + 4h HTF — Opening Range Breakout + Volume + HMA Trend

Hypothesis: 15m strategies failed (#085 Sharpe=-4.35, #089/#090/#093/#096 = 0 trades) because:
- Filters too strict (multiple confluence never aligned)
- Session filters blocked most opportunities
- Mean-reversion patterns whipsaw in trending crypto markets

Solution: Opening Range Breakout (ORB) is proven intraday pattern with natural trade limits:
- OR = first 4 bars of UTC day (1 hour at 15m)
- Breakout above/below OR in direction of 4h HMA trend
- Volume confirmation prevents false breakouts
- ONE trade per OR per direction (prevents churn)
- Very loose RSI filter (only blocks extreme counter-trend)
- Session weighting (prefer 00-12 UTC) but NOT hard filter

Key design:
- Timeframe: 15m (primary)
- HTF: 4h HMA(21) for trend bias (loaded ONCE before loop)
- Entry: ORB breakout + volume > 1.5x avg + HTF bias
- Position size: 0.20 (20% - conservative for 15m frequency)
- Stoploss: 2.0x ATR trailing
- Target: 50-100 trades/year, Sharpe > 0.167

Why this should work when #089 failed:
- ORB generates natural signal (breakout is objective, not subjective)
- Only 1 OR per day = max ~365 breakouts/year theoretically
- HTF + volume filter reduces to 50-100 actual trades
- No session hard-filter (just weighting) = more trades generate
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_orb_volume_hma_4h_v1"
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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m)
    
    # Opening Range tracking
    # OR = first 4 bars of each UTC day (00:00, 00:15, 00:30, 00:45)
    or_high = np.zeros(n)
    or_low = np.zeros(n)
    or_complete = np.zeros(n, dtype=bool)
    
    # Track which day we're on
    current_day = -1
    or_bar_count = 0
    day_or_high = 0.0
    day_or_low = float('inf')
    
    for i in range(n):
        # Extract UTC day from open_time (milliseconds timestamp)
        day = int(open_time[i] // (24 * 60 * 60 * 1000))
        
        if day != current_day:
            current_day = day
            or_bar_count = 0
            day_or_high = high[i]
            day_or_low = low[i]
        else:
            day_or_high = max(day_or_high, high[i])
            day_or_low = min(day_or_low, low[i])
        
        or_bar_count += 1
        
        if or_bar_count >= 4:
            or_complete[i] = True
            or_high[i] = day_or_high
            or_low[i] = day_or_low
        else:
            or_complete[i] = False
            or_high[i] = day_or_high
            or_low[i] = day_or_low
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track daily OR trades to prevent multiple entries on same OR
    traded_day_long = -1
    traded_day_short = -1
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if not or_complete[i]:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Extract current day and hour for tracking
        current_day_idx = int(open_time[i] // (24 * 60 * 60 * 1000))
        hour_utc = int((open_time[i] % (24 * 60 * 60 * 1000)) // (60 * 60 * 1000))
        prime_session = 0 <= hour_utc < 12
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === ORB BREAKOUT SIGNALS ===
        or_upper = or_high[i]
        or_lower = or_low[i]
        
        # Breakout = price crosses OR boundary (current bar breaks, previous didn't)
        breakout_bull = close[i] > or_upper and close[i-1] <= or_upper
        breakout_bear = close[i] < or_lower and close[i-1] >= or_lower
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.5 * vol_sma[i] if vol_sma[i] > 0 and not np.isnan(vol_sma[i]) else False
        
        # === RSI FILTER (LOOSE) ===
        rsi_ok_long = rsi[i] < 80.0
        rsi_ok_short = rsi[i] > 20.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 1.0
        
        # LONG: ORB breakout + HTF bull + volume + RSI ok + not already traded today
        if breakout_bull and htf_bull and vol_confirmed and rsi_ok_long and current_day_idx != traded_day_long:
            desired_signal = SIZE
            signal_strength = 1.0 if prime_session else 0.7
        # SHORT: ORB breakout + HTF bear + volume + RSI ok + not already traded today
        elif breakout_bear and htf_bear and vol_confirmed and rsi_ok_short and current_day_idx != traded_day_short:
            desired_signal = -SIZE
            signal_strength = 1.0 if prime_session else 0.7
        # Fallback: HMA confluence without ORB (smaller size, less frequent)
        elif htf_bull and hma_bull and rsi[i] < 65.0 and not in_position:
            desired_signal = SIZE * 0.5
        elif htf_bear and hma_bear and rsi[i] > 35.0 and not in_position:
            desired_signal = -SIZE * 0.5
        
        if desired_signal != 0.0:
            desired_signal *= signal_strength
        
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
        elif desired_signal >= SIZE * 0.4:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.4:
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
                # Mark day as traded for this direction
                if final_signal > 0:
                    traded_day_long = current_day_idx
                else:
                    traded_day_short = current_day_idx
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                if final_signal > 0:
                    traded_day_long = current_day_idx
                else:
                    traded_day_short = current_day_idx
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
        
        # Reset daily trade tracking at day boundary
        if current_day_idx != traded_day_long:
            traded_day_long = -1
        if current_day_idx != traded_day_short:
            traded_day_short = -1
        
        signals[i] = final_signal
    
    return signals