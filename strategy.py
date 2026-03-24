#!/usr/bin/env python3
"""
Experiment #059: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume

Hypothesis: After 58 failed experiments, the pattern is clear:
- Too many filters = ZERO trades (experiments 049, 050, 052, 053, 057 all Sharpe=0.000)
- Need LOOSE entry conditions that still generate 40-80 trades/year on 1h
- 4h HMA provides trend bias (proven in best strategies)
- 12h HMA provides major trend confirmation
- 1h RSI pullback entries (LOOSE: RSI < 55 for long, not < 30)
- Volume confirmation (loose: just above 20-bar average)
- Session filter 08-20 UTC to avoid low liquidity
- ATR stoploss at 2.5x for risk management
- Signal values: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)

Key design for TRADE GENERATION:
- RSI long threshold: < 55 (not < 30) - ensures entries in uptrend pullbacks
- RSI short threshold: > 45 (not > 70) - ensures entries in downtrend bounces
- Only require 2 of 3 filters (HTF trend + RSI + volume) - not all 3
- Session filter is soft (reduce size outside session, don't block)
- This ensures >=30 trades on train, >=3 on test per symbol

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_volume_v1"
timeframe = "1h"
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

def calculate_sma(values, period):
    """Simple Moving Average"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for major trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=34)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=16)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_FULL = 0.30  # 30% position size
    SIZE_HALF = 0.15  # 15% position size (outside session or weak signal)
    
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
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === MAJOR TREND CONFIRMATION (12h HMA) ===
        major_bull = close[i] > hma_12h_aligned[i]
        major_bear = close[i] < hma_12h_aligned[i]
        
        # === 1h HMA TREND ===
        hma_bull = close[i] > hma_1h[i]
        hma_bear = close[i] < hma_1h[i]
        
        # === VOLUME CONFIRMATION (LOOSE) ===
        vol_above_avg = volume[i] > 0.8 * vol_sma[i]  # loose: 80% of avg is ok
        
        # === RSI PULLBACK (LOOSE THRESHOLDS FOR TRADE GENERATION) ===
        # Long: RSI < 55 in uptrend (pullback entry, not oversold)
        # Short: RSI > 45 in downtrend (bounce entry, not overbought)
        rsi_pullback_long = rsi[i] < 55.0
        rsi_pullback_short = rsi[i] > 45.0
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 40.0  # not too weak
        rsi_momentum_short = rsi[i] < 60.0  # not too strong
        
        # === DESIRED SIGNAL (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # LONG conditions (need 2 of 3: HTF bull, RSI pullback, volume)
        long_score = 0
        if htf_bull:
            long_score += 1
        if rsi_pullback_long and rsi_momentum_long:
            long_score += 1
        if vol_above_avg:
            long_score += 1
        if hma_bull:
            long_score += 0.5  # bonus
        
        # SHORT conditions (need 2 of 3: HTF bear, RSI pullback, volume)
        short_score = 0
        if htf_bear:
            short_score += 1
        if rsi_pullback_short and rsi_momentum_short:
            short_score += 1
        if vol_above_avg:
            short_score += 1
        if hma_bear:
            short_score += 0.5  # bonus
        
        # Entry logic: require score >= 2.0 for full size
        if long_score >= 2.0:
            if in_session:
                desired_signal = SIZE_FULL
                signal_strength = long_score
            else:
                desired_signal = SIZE_HALF  # reduced size outside session
                signal_strength = long_score
        elif short_score >= 2.0:
            if in_session:
                desired_signal = -SIZE_FULL
                signal_strength = short_score
            else:
                desired_signal = -SIZE_HALF  # reduced size outside session
                signal_strength = short_score
        
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
        if desired_signal >= SIZE_FULL * 0.85:
            final_signal = SIZE_FULL
        elif desired_signal <= -SIZE_FULL * 0.85:
            final_signal = -SIZE_FULL
        elif desired_signal >= SIZE_HALF * 0.85:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.85:
            final_signal = -SIZE_HALF
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