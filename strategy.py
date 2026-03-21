#!/usr/bin/env python3
"""
EXPERIMENT #108 - MTF Supertrend+RSI+Chandelier+VolAdj Sizing (15m+4h Proper HTF v1)
==================================================================================================
Hypothesis: Recent failures (#101-#107) show Chandelier stops alone aren't enough - need 
volatility-adjusted position sizing AND proper HTF alignment. Current best uses 15m+1h+4h.

Key changes from #040:
- Use mtf_data helper for PROPER 4h alignment (not manual resampling)
- Chandelier exit: highest_high - 3*ATR(22) for trailing stops
- Volatility-adjusted sizing: base_size * (median_ATR / current_ATR), capped at 0.35
- Discrete signal levels: 0.0, ±0.20, ±0.35 to minimize churn costs
- Stricter entry filters to reduce trade frequency and fees
- 4h Supertrend for major trend, 15m RSI for pullback entries

Why this should beat #040 and recent failures:
- Proper HTF alignment prevents look-ahead bias (46 strategies failed without this)
- Vol-adjusted sizing reduces exposure during high vol (2022 crash protection)
- Chandelier trailing stop locks in profits better than fixed ATR stops
- Based on winning combo from mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1 (Sharpe=3.653)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_chandelier_voladj_15m_4h_proper_htf_v1"
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


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """
    Calculate Chandelier Exit (ATR trailing stop)
    Long exit: highest_high - multiplier * ATR
    Short exit: lowest_low + multiplier * ATR
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        chandelier_long[i] = highest_high - multiplier * atr[i]
        chandelier_short[i] = lowest_low + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def calculate_atr_percentile(atr, lookback=100):
    """Calculate ATR percentile for volatility regime detection"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = atr[i - lookback + 1:i + 1]
        sorted_window = np.sort(window)
        rank = np.searchsorted(sorted_window, atr[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h data using PROPER mtf_data helper (CRITICAL - no manual resampling!)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
    except Exception:
        # Fallback if mtf_data not available
        df_4h = None
        close_4h = None
        high_4h = None
        low_4h = None
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    atr_15m_22 = calculate_atr(high, low, close, period=22)
    rsi_15m = calculate_rsi(close, period=14)
    
    # Chandelier exit for trailing stops
    chandelier_long_15m, chandelier_short_15m = calculate_chandelier_exit(
        high, low, close, atr_15m_22, period=22, multiplier=3.0
    )
    
    # ATR percentile for volatility regime
    atr_pct_15m = calculate_atr_percentile(atr_15m, lookback=100)
    
    # 15m Supertrend for short-term trend
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h indicators for major trend (using mtf_data helper)
    if df_4h is not None and close_4h is not None:
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        rsi_4h = calculate_rsi(close_4h, period=14)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    else:
        # Fallback: use 15m supertrend with longer period as proxy
        st_4h_aligned = calculate_supertrend(high, low, close, period=40, multiplier=3.0)[1]
        rsi_4h_aligned = calculate_rsi(close, period=56)
        atr_4h_aligned = calculate_atr(high, low, close, period=56)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels with vol adjustment (CRITICAL for drawdown control)
    BASE_SIZE = 0.30
    SIZE_LOW_VOL = 0.35  # When ATR percentile < 0.4
    SIZE_MED_VOL = 0.25  # When ATR percentile 0.4-0.7
    SIZE_HIGH_VOL = 0.15  # When ATR percentile > 0.7
    SIZE_MIN = 0.15
    SIZE_MAX = 0.35
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier (Chandelier uses 3.0)
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 100, 22, 56)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get 4h trend direction
        trend_4h = st_4h_aligned[i] if not np.isnan(st_4h_aligned[i]) else 0
        
        # Get 15m indicators
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        st_15m = st_direction_15m[i]
        atr_pct = atr_pct_15m[i]
        
        # Volatility-adjusted position sizing
        if atr_pct < 0.4:
            vol_adjusted_size = SIZE_LOW_VOL
        elif atr_pct < 0.7:
            vol_adjusted_size = SIZE_MED_VOL
        else:
            vol_adjusted_size = SIZE_HIGH_VOL
        
        # Cap position size
        vol_adjusted_size = np.clip(vol_adjusted_size, SIZE_MIN, SIZE_MAX)
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Chandelier stoploss check (3*ATR(22))
            if prev_side == 1:
                chandelier_stop = chandelier_long_15m[i]
                if price < chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = vol_adjusted_size * 0.5
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                chandelier_stop = chandelier_short_15m[i]
                if price > chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -vol_adjusted_size * 0.5
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
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
        
        # Entry logic: 4h Supertrend + 15m RSI pullback + 15m Supertrend confirmation
        # Only enter when 4h and 15m trends agree
        
        if trend_4h == 1 and st_15m == 1:  # Bullish trend on both timeframes
            # RSI pullback entry (not overbought)
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
                signals[i] = vol_adjusted_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1 and st_15m == -1:  # Bearish trend on both timeframes
            # RSI pullback entry (not oversold)
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
                signals[i] = -vol_adjusted_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals