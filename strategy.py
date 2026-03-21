#!/usr/bin/env python3
"""
EXPERIMENT #111 - MTF HMA+Supertrend+RSI+BBW with Proper HTF Alignment (15m+4h v2)
==================================================================================================
Hypothesis: Experiments #101-#110 failed due to improper MTF resampling and overly aggressive
Chandelier stops. The current best (#040 equivalent) used 15m+1h but we should try 15m+4h
with PROPER mtf_data helper alignment (not manual resampling).

Key changes from #040:
- Use mtf_data helper (get_htf_data, align_htf_to_ltf) for TRUE 4h alignment
- 4h HMA for primary trend filter (more stable than 1h)
- 15m RSI for pullback entries (proven in #031, #034, #035)
- 15m Supertrend for stoploss (2.5*ATR for wider stops)
- BBW regime filter for position sizing (low vol = larger size, high vol = smaller)
- Position sizing: 0.25 base, scaled by volatility regime (0.20-0.35 range)
- Discrete signal levels to reduce churn costs
- Leverage: 1.0 (no leverage)

Why this should beat #040:
- Proper 4h alignment via mtf_data helper (no manual resampling bugs)
- 4h trend is more stable than 1h (fewer whipsaws)
- Volatility-adjusted sizing (larger positions in low vol regimes)
- Cleaner state tracking with fewer bugs
- Based on proven winning combinations from #031, #034, #035
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_supertrend_rsi_bbw_voladj_15m_4h_v2"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


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
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile rank for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i]) / len(window)
        percentile[i] = rank
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing and stoploss
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h trend filter using mtf_data helper (PROPER alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate 4h HMA for trend
        hma_4h_raw = calculate_hma(close_4h, period=21)
        hma_4h = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
        
        # Calculate 4h Supertrend for trend confirmation
        _, st_direction_4h_raw = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        st_direction_4h = align_htf_to_ltf(prices, df_4h, st_direction_4h_raw)
        
        # Calculate 4h BBW for regime
        _, _, _, bbw_4h_raw = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        bbw_4h = align_htf_to_ltf(prices, df_4h, bbw_4h_raw)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h = np.zeros(n)
        st_direction_4h = np.zeros(n)
        bbw_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on volatility regime (CRITICAL for drawdown control)
    # Low vol (BBW pct < 0.3) = larger size (0.35)
    # Medium vol (0.3-0.7) = medium size (0.25)
    # High vol (BBW pct > 0.7) = smaller size (0.20)
    SIZE_HIGH_VOL = 0.20
    SIZE_MED_VOL = 0.25
    SIZE_LOW_VOL = 0.35
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier (wider than #040 to avoid premature exits)
    ATR_STOP_MULT = 2.5
    
    # 4h BBW minimum for regime filter
    BBW_4H_MIN = 0.01
    
    first_valid = max(200, 14 * 2, 20, 100)
    
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
        
        # 4h trend filters
        trend_4h = 0
        if hma_4h[i] > 0:
            if close[i] > hma_4h[i]:
                trend_4h = 1
            elif close[i] < hma_4h[i]:
                trend_4h = -1
        
        st_trend_4h = st_direction_4h[i] if st_direction_4h[i] != 0 else 0
        bbw_4h_val = bbw_4h[i] if bbw_4h[i] > 0 else bbw_15m[i]
        
        # 15m Supertrend for entry confirmation
        st_15m = st_direction_15m[i]
        
        # Volatility regime for position sizing
        bbw_pct = bbw_pct_15m[i]
        if bbw_pct < 0.3:
            position_size = SIZE_LOW_VOL
        elif bbw_pct < 0.7:
            position_size = SIZE_MED_VOL
        else:
            position_size = SIZE_HIGH_VOL
        
        # 4h BBW filter - avoid extremely choppy markets
        if bbw_4h_val < BBW_4H_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
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
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = position_size / 2
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
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
                continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
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
                    signals[i] = -position_size / 2
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
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
                continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        
        # Entry logic: 4h HMA + Supertrend + 15m Supertrend + RSI pullback
        if trend_4h == 1 and st_trend_4h == 1 and st_15m == 1:  # Bullish trend confirmed
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:  # Pullback entry
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1 and st_trend_4h == -1 and st_15m == -1:  # Bearish trend confirmed
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:  # Pullback entry
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals