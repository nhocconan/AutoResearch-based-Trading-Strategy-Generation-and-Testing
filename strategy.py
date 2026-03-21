#!/usr/bin/env python3
"""
EXPERIMENT #011 - MTF EMA+Stoch+ADX+Volume (15m+1h+4h v1)
==================================================================================================
Hypothesis: Replace HMA/KAMA with simpler EMA crossover (21/55) for more stable trend detection.
Use Stochastic (14,3,3) instead of RSI for entry timing - different signal characteristics.
ADX(14) on 15m filters choppy markets. Volume spike confirms breakout momentum.

Why this should work:
- EMA crossover is smoother and less prone to whipsaws than HMA
- Stochastic provides earlier entry signals than RSI (more sensitive)
- ADX > 20 ensures we only trade in trending markets
- Volume confirmation reduces false breakouts
- Three timeframes: 15m base, 1h momentum, 4h trend

Key differences from current best (#009):
- EMA instead of KAMA for trend
- Stochastic instead of RSI for entries
- Volume confirmation added
- Simpler indicator stack (may be more robust)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_ema_stoch_adx_volume_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = pd.Series(close).ewm(span=period, adjust=False).mean().values
    return ema


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
    n = len(close)
    if n < k_period:
        return np.zeros(n), np.zeros(n)
    
    lowest_low = pd.Series(low).rolling(window=k_period, min_periods=k_period).min().values
    highest_high = pd.Series(high).rolling(window=k_period, min_periods=k_period).max().values
    
    k = np.zeros(n)
    for i in range(n):
        if highest_high[i] > lowest_low[i]:
            k[i] = 100 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
        else:
            k[i] = 50
    
    d = pd.Series(k).ewm(span=d_period, adjust=False).mean().values
    
    return k, d


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(n):
        if (plus_di[i] + minus_di[i]) > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    
    return adx


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume spike detection"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    stoch_k_15m, stoch_d_15m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    adx_15m = calculate_adx(high, low, close, period=14)
    vol_sma_15m = calculate_volume_sma(volume, period=20)
    
    # Get 1h data using mtf_data helper
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        # 1h Stochastic for momentum
        stoch_k_1h, stoch_d_1h = calculate_stochastic(h_1h, l_1h, c_1h, k_period=14, d_period=3)
        
        # Align 1h indicators to 15m timeframe
        stoch_k_1h_aligned = align_htf_to_ltf(prices, df_1h, stoch_k_1h)
        stoch_d_1h_aligned = align_htf_to_ltf(prices, df_1h, stoch_d_1h)
    except Exception:
        stoch_k_1h_aligned = np.full(n, 50.0)
        stoch_d_1h_aligned = np.full(n, 50.0)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        
        # 4h EMA crossover for trend
        ema_21_4h = calculate_ema(c_4h, period=21)
        ema_55_4h = calculate_ema(c_4h, period=55)
        
        # Align 4h indicators to 15m timeframe
        ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
        ema_55_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_55_4h)
    except Exception:
        ema_21_4h_aligned = np.zeros(n)
        ema_55_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # Stochastic thresholds for entries
    STOCH_LONG_ENTRY = 30  # %K crosses above from oversold
    STOCH_SHORT_ENTRY = 70  # %K crosses below from overbought
    
    # ADX minimum for trending market
    ADX_MIN = 20
    
    # Volume spike threshold
    VOL_SPIKE_MULT = 1.5
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 55, 14 * 2, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(adx_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get aligned MTF values
        ema_21_4h = ema_21_4h_aligned[i] if i < len(ema_21_4h_aligned) else 0
        ema_55_4h = ema_55_4h_aligned[i] if i < len(ema_55_4h_aligned) else 0
        stoch_k_1h = stoch_k_1h_aligned[i] if i < len(stoch_k_1h_aligned) else 50
        stoch_d_1h = stoch_d_1h_aligned[i] if i < len(stoch_d_1h_aligned) else 50
        
        # ADX filter - only trade in trending markets
        if adx_15m[i] < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h trend filter (EMA crossover)
        if ema_21_4h > 0 and ema_55_4h > 0:
            if ema_21_4h > ema_55_4h:
                trend_4h = 1  # Bullish
            elif ema_21_4h < ema_55_4h:
                trend_4h = -1  # Bearish
            else:
                trend_4h = 0  # Neutral
        else:
            trend_4h = 0
        
        if trend_4h == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check existing positions first (stoploss/TP management)
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            
            # Initialize highest/lowest if needed
            if highest_since_entry[i - 1] == 0:
                highest_since_entry[i - 1] = prev_entry
            if lowest_since_entry[i - 1] == 0:
                lowest_since_entry[i - 1] = prev_entry
            
            # Update highest/lowest since entry
            current_high = max(highest_since_entry[i - 1], close[i])
            current_low = min(lowest_since_entry[i - 1], close[i])
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 1h Stochastic + 15m Stochastic + Volume + ADX
        # Volume spike confirmation
        vol_spike = volume[i] > (VOL_SPIKE_MULT * vol_sma_15m[i]) if vol_sma_15m[i] > 0 else False
        
        if trend_4h == 1:  # Bullish trend on 4h
            # 1h Stochastic confirmation (momentum)
            # 15m Stochastic pullback entry (from oversold)
            # Volume confirmation
            if (stoch_k_1h > 40 and stoch_d_1h > 40 and  # 1h momentum positive
                stoch_k_15m[i] < STOCH_LONG_ENTRY + 20 and stoch_k_15m[i] > STOCH_LONG_ENTRY - 10 and
                stoch_k_15m[i] > stoch_d_15m[i] and  # %K above %D (bullish cross)
                vol_spike):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = False
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend_4h == -1:  # Bearish trend on 4h
            # 1h Stochastic confirmation (momentum)
            # 15m Stochastic pullback entry (from overbought)
            # Volume confirmation
            if (stoch_k_1h < 60 and stoch_d_1h < 60 and  # 1h momentum negative
                stoch_k_15m[i] > STOCH_SHORT_ENTRY - 20 and stoch_k_15m[i] < STOCH_SHORT_ENTRY + 10 and
                stoch_k_15m[i] < stoch_d_15m[i] and  # %K below %D (bearish cross)
                vol_spike):
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = False
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals