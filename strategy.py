#!/usr/bin/env python3
"""
Experiment #065: 15m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m strategies have failed (Sharpe=0.000) due to ZERO TRADES from overly strict filters.
This strategy uses LOOSE entry conditions to ensure trades generate on ALL symbols:
- 15m HMA(21) for short-term trend direction
- 4h HMA(50) for medium-term bias (ONE agreement needed, not all HTF)
- RSI(7) pullback entries (oversold in uptrend, overbought in downtrend)
- Session filter: 00-12 UTC (London+NY overlap = cleaner moves, less noise)
- Choppiness Index regime: CHOP<50 = trend follow, CHOP>50 = mean revert
- Position size: 0.18 (18% of capital, smaller for 15m frequency)
- Stoploss: 2.0x ATR trailing (tighter for 15m swings)

Key design to AVOID zero trades:
- RSI thresholds LOOSE: long when RSI<50 (not <30), short when RSI>50 (not >70)
- HTF bias: only need 4h OR 1d agreement (not both)
- Session filter reduces noise but doesn't block all entries
- Fallback: pure 15m HMA crossover if HTF unclear

Target: Sharpe>0.167, DD>-40%, trades>=40 on train, trades>=5 on test, ALL symbols Sharpe>0
Trade frequency: 50-100/year (0.10-0.20% daily fee drag acceptable)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_4h1d_session_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (smaller for 15m frequency)
    
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
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        is_prime_session = 0 <= hour_utc <= 12  # London+NY overlap
        
        # === HTF BIAS (4h and 1d HMA) ===
        # Only need ONE HTF agreement (not both) to avoid blocking trades
        htf_4h_bull = not np.isnan(hma_4h_aligned[i]) and close[i] > hma_4h_aligned[i]
        htf_4h_bear = not np.isnan(hma_4h_aligned[i]) and close[i] < hma_4h_aligned[i]
        htf_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # At least one HTF agrees (loose filter)
        htf_bull = htf_4h_bull or htf_1d_bull
        htf_bear = htf_4h_bear or htf_1d_bear
        htf_neutral = not htf_bull and not htf_bear
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trending = chop[i] < 50.0
        is_choppy = chop[i] >= 50.0
        
        # === RSI PULLBACK SIGNALS (LOOSE thresholds) ===
        # Long: RSI < 50 (not <30) in uptrend
        # Short: RSI > 50 (not >70) in downtrend
        rsi_long = rsi[i] < 50.0
        rsi_short = rsi[i] > 50.0
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow HTF direction with RSI pullback
            # LONG: HTF bull + 15m HMA bull + RSI pullback
            if htf_bull and hma_bull and rsi_long:
                if is_prime_session:
                    desired_signal = SIZE
                else:
                    desired_signal = SIZE * 0.7  # Reduce size outside prime session
            # SHORT: HTF bear + 15m HMA bear + RSI pullback
            elif htf_bear and hma_bear and rsi_short:
                if is_prime_session:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.7
            # Fallback: 15m HMA only (if HTF neutral)
            elif htf_neutral and hma_bull and rsi_oversold:
                desired_signal = SIZE * 0.5
            elif htf_neutral and hma_bear and rsi_overbought:
                desired_signal = -SIZE * 0.5
        else:
            # CHOPPY REGIME: Mean reversion at extremes
            # LONG: RSI very oversold + 15m HMA not strongly bear
            if rsi_oversold and not hma_bear:
                if is_prime_session:
                    desired_signal = SIZE * 0.8
                else:
                    desired_signal = SIZE * 0.5
            # SHORT: RSI very overbought + 15m HMA not strongly bull
            elif rsi_overbought and not hma_bull:
                if is_prime_session:
                    desired_signal = -SIZE * 0.8
                else:
                    desired_signal = -SIZE * 0.5
        
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