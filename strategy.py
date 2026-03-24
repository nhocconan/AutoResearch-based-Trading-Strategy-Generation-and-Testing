#!/usr/bin/env python3
"""
Experiment #093: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe requires EXTREME selectivity to overcome fee drag.
- Session filter (08-20 UTC) captures London/NY overlap = 70% of daily volume
- 4h HMA provides major trend bias (NEVER trade counter-trend on 5m)
- 15m HMA provides intermediate trend confirmation
- 5m RSI pullback (30-70 range) for entry timing within HTF trend
- Choppiness Index filters out choppy periods (only trade when CHOP<50)
- This combines: HTF trend (proven) + session filter (liquidity) + RSI pullback (timing)

Key design choices:
- Timeframe: 5m (FIRST 5m experiment — extreme care needed)
- HTF: 4h HMA (major trend) + 15m HMA (intermediate confirmation)
- Session: 08:00-20:00 UTC only (avoid Asian low-liquidity + weekend)
- Entry: RSI pullback to 40-60 zone IN DIRECTION of HTF trend
- Regime: CHOP<50 = trending (trade), CHOP>50 = choppy (stand aside)
- Position size: 0.15 (15% — conservative for 5m fee drag)
- Stoploss: 2.0x ATR trailing (tighter for 5m volatility)
- Target: 50-120 trades/year, Sharpe>0.167 (beat current best)

Why this might work on 5m:
- Session filter removes 60% of bars = fewer trades, better quality
- Dual HTF (15m+4h) ensures we only trade with established trend
- RSI pullback (not breakout) = better risk/reward on lower TF
- CHOP filter avoids whipsaw during range-bound periods
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_hma_rsi_chop_15m4h_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    We use threshold 50.0 for 5m (stricter)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_momentum(close, period=10):
    """Rate of Change momentum"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    mom = np.zeros(n)
    mom[:] = np.nan
    for i in range(period, n):
        if close[i-period] > 1e-10:
            mom[i] = (close[i] - close[i-period]) / close[i-period] * 100.0
        else:
            mom[i] = 0.0
    
    return mom

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMA for trend bias
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    momentum = calculate_momentum(close, period=10)
    
    signals = np.zeros(n)
    SIZE = 0.15  # 15% position size (conservative for 5m fee drag)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Extract hour from open_time for session filter
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_5m[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
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
        if np.isnan(momentum[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC only) ===
        # London open (08:00) to NY close (20:00) = highest liquidity
        in_session = (hours[i] >= 8) and (hours[i] < 20)
        
        if not in_session:
            # Close positions outside session
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h + 15m HMA) ===
        # BOTH must agree for strong trend signal
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # Strong trend: both HTF agree
        strong_bull = htf_4h_bull and htf_15m_bull
        strong_bear = htf_4h_bear and htf_15m_bear
        
        # === REGIME FILTER (Choppiness Index) ===
        # Only trade when CHOP < 50 (trending market)
        is_trending = chop[i] < 50.0
        
        if not is_trending:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === RSI PULLBACK ENTRY (within trend) ===
        # Long: RSI pulled back to 35-55 zone in bull trend
        # Short: RSI pulled back to 45-65 zone in bear trend
        rsi_pullback_long = (rsi[i] >= 35.0) and (rsi[i] <= 55.0)
        rsi_pullback_short = (rsi[i] >= 45.0) and (rsi[i] <= 65.0)
        
        # RSI confirmation (momentum in right direction)
        rsi_momentum_long = rsi[i] > 40.0
        rsi_momentum_short = rsi[i] < 60.0
        
        # === MOMENTUM CONFIRMATION ===
        # Price momentum should align with trend
        mom_long = momentum[i] > -0.5  # not strongly negative
        mom_short = momentum[i] < 0.5  # not strongly positive
        
        # === 5m HMA CONFIRMATION ===
        hma_5m_bull = close[i] > hma_5m[i]
        hma_5m_bear = close[i] < hma_5m[i]
        
        # === DESIRED SIGNAL (Multi-confluence) ===
        desired_signal = 0.0
        
        # LONG: strong bull HTF + RSI pullback + momentum + 5m HMA
        if strong_bull and rsi_pullback_long and rsi_momentum_long and mom_long and hma_5m_bull:
            desired_signal = SIZE
        
        # SHORT: strong bear HTF + RSI pullback + momentum + 5m HMA
        elif strong_bear and rsi_pullback_short and rsi_momentum_short and mom_short and hma_5m_bear:
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