#!/usr/bin/env python3
"""
EXPERIMENT #041 - MTF DEMA+Supertrend+RSI+ATR Dynamic Sizing (15m+4h Clean v2)
==================================================================================================
Hypothesis: Current #040 uses complex MTF resampling that may have bugs. 
Using mtf_data helper properly should fix alignment issues and improve Sharpe.

Key changes from #040:
- Use mtf_data.get_htf_data() and align_htf_to_ltf() for PROPER 4h alignment
- DEMA(8/21) instead of HMA for faster trend detection (less lag)
- Dynamic position sizing: base_size * (target_vol / current_vol)
- Simpler state tracking to avoid index errors that crashed #027-#030
- RSI thresholds: 35-65 (wider than #040's 40-60 for more entries)
- ATR stoploss: 2.5*ATR (slightly wider than #040's 2.0*ATR)
- Position size: 0.30 base (slightly lower than #040's 0.35 for safety)

Why this should beat #040:
- Proper MTF alignment using mtf_data helper (no manual resampling bugs)
- DEMA reacts faster to trend changes than HMA
- Dynamic sizing reduces position when volatility spikes (better DD control)
- Simpler logic = fewer edge case crashes
- Based on #025 which had positive Sharpe with similar logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_dema_supertrend_rsi_atr_dynamic_15m_4h_v2"
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


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean().values
    
    dema = 2 * ema1 - ema2
    dema[:period] = np.nan
    
    return dema


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    middle = rolling_mean
    upper = middle + std_mult * rolling_std
    lower = middle - std_mult * rolling_std
    
    bbw = np.zeros(n)
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return upper, middle, lower, bbw


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    dema_fast_15m = calculate_dema(close, period=8)
    dema_slow_15m = calculate_dema(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Get 4h data using mtf_data helper (PROPER alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend
        dema_fast_4h = calculate_dema(close_4h, period=8)
        dema_slow_4h = calculate_dema(close_4h, period=21)
        supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        dema_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, dema_fast_4h)
        dema_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, dema_slow_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        dema_fast_4h_aligned = np.zeros(n)
        dema_slow_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
        bbw_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries (wider range for more entries)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # BBW minimum for regime filter (4h)
    BBW_MIN = 0.015
    
    # Volatility target for dynamic sizing
    TARGET_VOL = 0.02  # 2% daily vol target
    
    first_valid = max(200, 14 * 2, 21, 28)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for NaN or zero ATR
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h trend filters
        trend_4h = 0
        if dema_fast_4h_aligned[i] > dema_slow_4h_aligned[i]:
            trend_4h = 1
        elif dema_fast_4h_aligned[i] < dema_slow_4h_aligned[i]:
            trend_4h = -1
        
        st_trend_4h = st_direction_4h_aligned[i]
        bbw_4h_val = bbw_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        
        # BBW filter - avoid choppy markets (4h)
        if bbw_4h_val < BBW_MIN or np.isnan(bbw_4h_val):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (DEMA + Supertrend on 4h)
        if trend_4h != st_trend_4h or trend_4h == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Calculate dynamic position size based on volatility
        current_vol = atr_15m[i] / close[i] * 16  # Approximate daily vol from 15m ATR
        if current_vol > 0:
            vol_scalar = min(1.5, max(0.5, TARGET_VOL / current_vol))
        else:
            vol_scalar = 1.0
        
        position_size = BASE_SIZE * vol_scalar
        position_size = min(0.35, max(0.15, position_size))  # Clamp to safe range
        position_size_half = position_size / 2
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            price = close[i]
            atr = atr_15m[i]
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = position_size_half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -position_size_half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
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
        
        # Entry logic: 4h DEMA + Supertrend + BBW + 15m RSI
        price = close[i]
        rsi_val = rsi_15m[i]
        
        if trend_4h == 1 and st_trend_4h == 1:  # Bullish trend confirmed on 4h
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:  # Pullback entry
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1 and st_trend_4h == -1:  # Bearish trend confirmed on 4h
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:  # Pullback entry
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals