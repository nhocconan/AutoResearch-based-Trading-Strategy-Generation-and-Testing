#!/usr/bin/env python3
"""
Experiment #845: 15m Primary + 4h/1d HTF — Intraday Momentum with HTF Bias

Hypothesis: 15m timeframe can work with LOOSE entry conditions + HTF bias for direction.
Previous 15m experiments failed due to overly strict filters (0 trades). This version:
1. Uses 4h HMA(21) for trend BIAS (not hard filter) - increases conviction
2. Uses 1d HMA(21) for major regime direction
3. Uses 15m RSI(7) with loose thresholds (35/65 not 20/80) for entries
4. Uses 15m HMA(8/21) crossover for entry timing precision
5. Volume filter: only trade when volume > 0.5x 20-bar average
6. Session filter: 00-12 UTC preferred (London/NY overlap) but not blocking
7. Size: 0.15 base, 0.25 with HTF confluence (smaller for 15m frequency)
8. Stoploss: 2.0x ATR(14) trailing

Target: Sharpe>0.40, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_volume_4h1d_v1"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
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
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    hma_8 = calculate_hma(close, period=8)
    hma_21 = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA for filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]) or np.isnan(rsi_7[i]):
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
        
        # === SESSION FILTER (00-12 UTC preferred, but not blocking) ===
        hour = (open_time[i] // 3600000) % 24
        is_prime_session = 0 <= hour < 12
        
        # === VOLUME FILTER (avoid low liquidity) ===
        volume_ok = volume[i] > 0.5 * vol_ma20[i] if not np.isnan(vol_ma20[i]) else True
        
        # === HTF BIAS (4h HMA) - soft filter for conviction ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === MAJOR TREND (1d HMA) - regime direction ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 15m HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_8[i-1]) and not np.isnan(hma_21[i-1]):
            hma_cross_long = (hma_8[i-1] <= hma_21[i-1]) and (hma_8[i] > hma_21[i])
            hma_cross_short = (hma_8[i-1] >= hma_21[i-1]) and (hma_8[i] < hma_21[i])
        
        # === 15m HMA TREND ===
        hma_15m_bull = hma_8[i] > hma_21[i]
        hma_15m_bear = hma_8[i] < hma_21[i]
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_oversold = rsi_7[i] < 40.0
        rsi_overbought = rsi_7[i] > 60.0
        rsi_extreme_oversold = rsi_7[i] < 30.0
        rsi_extreme_overbought = rsi_7[i] > 70.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        htf_confluence = False
        
        # LONG: Multiple paths to entry
        if hma_cross_long or (hma_15m_bull and rsi_oversold):
            # Check HTF alignment for conviction
            if htf_4h_bull and htf_1d_bull:
                htf_confluence = True
            
            if rsi_extreme_oversold or hma_cross_long:
                desired_signal = SIZE_STRONG if htf_confluence else SIZE_BASE
            elif volume_ok and is_prime_session:
                desired_signal = SIZE_BASE
        
        # SHORT: Multiple paths to entry
        elif hma_cross_short or (hma_15m_bear and rsi_overbought):
            # Check HTF alignment for conviction
            if htf_4h_bear and htf_1d_bear:
                htf_confluence = True
            
            if rsi_extreme_overbought or hma_cross_short:
                desired_signal = -SIZE_STRONG if htf_confluence else -SIZE_BASE
            elif volume_ok and is_prime_session:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                entry_atr = atr_14[i]
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