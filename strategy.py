#!/usr/bin/env python3
"""
EXPERIMENT #030 - MTF HMA+KAMA+Stoch+RSI+BBW (15m+4h Clean v2)
==================================================================================================
Hypothesis: Based on #021 (Sharpe=4.629) and #023 (Sharpe=3.981), 15m entries with MTF filters work best.
Key changes from #039:
- Timeframe: 15m (proven in #019, #021, #023, #024 with Sharpe > 3.5)
- MTF: 15m + 4h (more stable trend filter than 1h)
- Proper resampling using open_time index (NOT pd.date_range - see #029 failure)
- Simplified entry logic with fewer conflicting signals
- Position size: 0.35 max (proven safe)
- Add stochastic for momentum confirmation (worked in #021, #023)
- BBW regime filter to avoid choppy markets
- ATR-based stoploss at 2.5*ATR (slightly looser for better win rate)

Why this should beat #021:
- 4h trend filter is more stable than 1h (fewer whipsaws)
- Cleaner MTF resampling using open_time
- Stochastic + RSI combination for better entry timing
- Based on proven winning combinations from #021, #023, #024
"""

import numpy as np
import pandas as pd

name = "mtf_hma_kama_stoch_rsi_bbw_15m_4h_v2"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    n = len(close)
    if n < k_period:
        return np.zeros(n), np.zeros(n)
    
    stoch_k = np.zeros(n)
    stoch_d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest = np.min(low[i - k_period + 1:i + 1])
        highest = np.max(high[i - k_period + 1:i + 1])
        
        if highest > lowest:
            stoch_k[i] = 100 * (close[i] - lowest) / (highest - lowest)
        else:
            stoch_k[i] = 50
    
    for i in range(k_period - 1 + d_period - 1, n):
        stoch_d[i] = np.mean(stoch_k[i - d_period + 1:i + 1])
    
    return stoch_k, stoch_d


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Check if open_time exists for proper resampling
    if 'open_time' in prices.columns:
        # Proper MTF resampling using open_time (NOT pd.date_range!)
        prices_indexed = prices.set_index('open_time')
        
        # Resample to 4h for trend filter
        df_4h = prices_indexed.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate 4h indicators
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        n_4h = len(c_4h)
        
        hma_4h = calculate_hma(c_4h, period=21)
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        
        # Map 4h indicators back to 15m using reindex with ffill
        trend_4h_series = pd.Series(np.where(c_4h > hma_4h, 1, np.where(c_4h < hma_4h, -1, 0)), index=df_4h.index)
        kama_trend_4h_series = pd.Series(np.where(c_4h > kama_4h, 1, np.where(c_4h < kama_4h, -1, 0)), index=df_4h.index)
        bbw_4h_series = pd.Series(bbw_4h, index=df_4h.index)
        
        # Reindex to 15m timeframe with forward fill
        trend_4h = trend_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
        kama_trend_4h = kama_trend_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
        bbw_4h_mapped = bbw_4h_series.reindex(prices_indexed.index, method='ffill').fillna(0).values
    else:
        # Fallback: simple downsampling if open_time not available
        bars_per_4h = 16  # 16 x 15m = 4h
        n_4h = n // bars_per_4h
        
        c_4h = np.zeros(n_4h)
        h_4h = np.zeros(n_4h)
        l_4h = np.zeros(n_4h)
        
        for i in range(n_4h):
            start_idx = i * bars_per_4h
            end_idx = start_idx + bars_per_4h
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
        
        hma_4h = calculate_hma(c_4h, period=21)
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        
        trend_4h = np.zeros(n)
        kama_trend_4h = np.zeros(n)
        bbw_4h_mapped = np.zeros(n)
        
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h < n_4h and idx_4h >= 40:
                if c_4h[idx_4h] > hma_4h[idx_4h]:
                    trend_4h[i] = 1
                elif c_4h[idx_4h] < hma_4h[idx_4h]:
                    trend_4h[i] = -1
                
                if c_4h[idx_4h] > kama_4h[idx_4h]:
                    kama_trend_4h[i] = 1
                elif c_4h[idx_4h] < kama_4h[idx_4h]:
                    kama_trend_4h[i] = -1
                
                bbw_4h_mapped[i] = bbw_4h[idx_4h]
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    stoch_k_15m, stoch_d_15m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    hma_15m = calculate_hma(close, period=21)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    
    # Stochastic thresholds
    STOC_LONG_MIN = 40
    STOC_LONG_MAX = 70
    STOC_SHORT_MIN = 30
    STOC_SHORT_MAX = 60
    
    # BBW minimum for regime filter (avoid choppy markets)
    BBW_MIN = 0.015
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 40 * 16, 14 * 2, 20, 28)
    
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
        
        trend = trend_4h[i]
        kama_trend = kama_trend_4h[i]
        rsi_val = rsi_15m[i]
        stoch_k = stoch_k_15m[i]
        stoch_d = stoch_d_15m[i]
        atr = atr_15m[i]
        price = close[i]
        bbw_4h_val = bbw_4h_mapped[i]
        bbw_15m_val = bbw_15m[i]
        
        # BBW filter - avoid choppy markets (4h and 15m)
        if bbw_4h_val < BBW_MIN or bbw_15m_val < BBW_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (HMA + KAMA on 4h)
        if trend != kama_trend or trend == 0:
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
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
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
                    signals[i] = SIZE_HALF
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
                    signals[i] = -SIZE_HALF
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
        
        # Entry logic: 4h HMA + KAMA + BBW + 15m RSI + Stochastic
        if trend == 1 and kama_trend == 1:  # Bullish trend confirmed on 4h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                STOC_LONG_MIN <= stoch_k <= STOC_LONG_MAX and
                stoch_k > stoch_d):  # Pullback + momentum confirmation
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and kama_trend == -1:  # Bearish trend confirmed on 4h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                STOC_SHORT_MIN <= stoch_k <= STOC_SHORT_MAX and
                stoch_k < stoch_d):  # Pullback + momentum confirmation
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals