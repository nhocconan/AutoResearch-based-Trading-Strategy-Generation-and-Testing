#!/usr/bin/env python3
"""
Experiment #687: 6h Primary + 1d HTF — Donchian Breakout + Volume Confirm + ATR Stop

Hypothesis: 6h timeframe offers unique balance between 4h (too noisy) and 12h (too slow).
Donchian(20) breakouts capture multi-day momentum moves. 1d HMA(21) provides HTF bias
without being overly restrictive (unlike requiring both 1d+1w alignment which killed trades).
Volume spike confirmation filters false breakouts. ATR(14) trailing stop manages risk.

Key innovations:
1. Donchian(20) breakout - clean 20-bar high/low break (4-5 days on 6h)
2. 1d HMA(21) bias - only long when price > daily HMA, only short when below
3. Volume spike filter - breakout must have volume > 1.5x 20-bar avg (confirms conviction)
4. ATR(14) trailing stop - 2.5x for risk management, signal→0 on stop
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
6. NO 1w filter - too restrictive, caused 0 trades in previous 6h experiments

Entry conditions (LOOSE to ensure trades):
- LONG: price > 1d HMA AND price breaks Donchian(20) high AND volume > 1.5x avg
- SHORT: price < 1d HMA AND price breaks Donchian(20) low AND volume > 1.5x avg
- No RSI/CHOP filters (these caused 0 trades in #675, #680, #683)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_breakout_volume_1d_v1"
timeframe = "6h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Volume ratio - current volume vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio[:period] = np.nan
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track breakout signals to avoid repeated entries
    last_long_signal = -100
    last_short_signal = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        # Price breaks above upper band (new 20-bar high)
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        # Price breaks below lower band (new 20-bar low)
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + Donchian breakout + volume confirm
        if htf_1d_bull and breakout_long and volume_confirmed:
            # Cooldown: don't re-enter within 5 bars of last long signal
            if i - last_long_signal > 5:
                desired_signal = SIZE_STRONG
                last_long_signal = i
        
        # SHORT: 1d bearish + Donchian breakout + volume confirm
        elif htf_1d_bear and breakout_short and volume_confirmed:
            # Cooldown: don't re-enter within 5 bars of last short signal
            if i - last_short_signal > 5:
                desired_signal = -SIZE_STRONG
                last_short_signal = i
        
        # Weaker entry: breakout without volume (smaller size)
        elif htf_1d_bull and breakout_long:
            if i - last_long_signal > 5:
                desired_signal = SIZE_BASE
                last_long_signal = i
        
        elif htf_1d_bear and breakout_short:
            if i - last_short_signal > 5:
                desired_signal = -SIZE_BASE
                last_short_signal = i
        
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