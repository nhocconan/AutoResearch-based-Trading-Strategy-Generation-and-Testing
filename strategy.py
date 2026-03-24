#!/usr/bin/env python3
"""
Experiment #345: 15m Primary + 4h/1d HTF — Session-Aware HMA/RSI Mean Reversion

Hypothesis: 15m strategies failed (Sharpe=0.000) because entry conditions were TOO STRICT.
This version LOOSENS entries while maintaining selectivity via HTF alignment + session filter.

Key innovations:
1. 4h HMA(21) for trend bias (direction filter)
2. 1d HMA(50) for regime (bull/bear market context)
3. 15m RSI(7) for entry timing — LOOSENED thresholds (25/75 instead of 20/80)
4. Session filter: prefer 00-12 UTC (London/NY overlap for crypto liquidity)
5. Choppiness Index: avoid entering when CHOP > 65 (too choppy)
6. Volume confirmation: require volume > 0.8 * 20-bar avg (avoid low-liquidity traps)
7. ATR(14) stoploss at 2.5x from entry

Position sizing: 0.15 base, 0.25 when HTF fully aligned (discrete levels)
Target: 50-100 trades/year, Sharpe > 0.40, DD > -40%

CRITICAL: Entry conditions LOOSENED to ensure trades generate (previous 15m = 0 trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_hma_rsi_chop_vol_4h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    sma_200 = calculate_sma(close, 200)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        preferred_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === CHOPPINESS FILTER (avoid choppy markets) ===
        chop_ok = chop[i] < 65.0  # Avoid very choppy conditions
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > 0.8 * vol_sma[i]  # At least 80% of avg volume
        
        # === HTF BIAS (4h and 1d) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES (LOOSENED for more trades) ===
        rsi_oversold = rsi[i] < 30.0  # Was 25, loosened
        rsi_overbought = rsi[i] > 70.0  # Was 75, loosened
        
        # === RSI DIVERGENCE CHECK (simplified) ===
        rsi_rising = False
        rsi_falling = False
        if i >= 3 and not np.isnan(rsi[i-1]) and not np.isnan(rsi[i-2]):
            rsi_rising = rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]
            rsi_falling = rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: RSI oversold + HTF bull + trend confirmation
        # Loosened: only need 4h OR 1d bull (not both)
        if rsi_oversold and chop_ok and vol_ok:
            if htf_4h_bull and hma_bull and above_sma200:
                # Strong long: 4h + 15m trend + SMA200
                if preferred_session:
                    desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
                else:
                    desired_signal = SIZE_BASE if htf_1d_bull else SIZE_BASE * 0.5
            elif htf_1d_bull and hma_bull:
                # Moderate long: 1d + 15m trend (4h neutral)
                if preferred_session:
                    desired_signal = SIZE_BASE
                elif rsi_rising:
                    desired_signal = SIZE_BASE * 0.5
        
        # SHORT ENTRY: RSI overbought + HTF bear + trend confirmation
        if rsi_overbought and chop_ok and vol_ok:
            if htf_4h_bear and hma_bear and below_sma200:
                # Strong short: 4h + 15m trend + SMA200
                if preferred_session:
                    desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
                else:
                    desired_signal = -SIZE_BASE if htf_1d_bear else -SIZE_BASE * 0.5
            elif htf_1d_bear and hma_bear:
                # Moderate short: 1d + 15m trend (4h neutral)
                if preferred_session:
                    desired_signal = -SIZE_BASE
                elif rsi_falling:
                    desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals