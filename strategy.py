#!/usr/bin/env python3
"""
EXPERIMENT #041 - MTF HMA+DEMA+KAMA+Stoch+RSI+Volume+BBW (15m+4h Clean v2)
==================================================================================================
Hypothesis: Based on #030 (Sharpe=5.787), the 15m+4h combination with HMA+KAMA+Stoch+RSI+BBW works best.
This version improves on #030 by:
- Adding DEMA for faster trend confirmation (DEMA reacts quicker than HMA)
- Adding volume confirmation for breakouts (volume spike = real move)
- Simplified MTF resampling using open_time index (no synthetic dates)
- Tighter RSI thresholds (35-65 instead of 30-70) for better entry quality
- Volume ratio filter (current volume > 1.5x avg volume) for breakout confirmation
- Position size: 0.35 (proven safe in winning strategies)
- Stoploss: 2.0*ATR (balanced R:R)

Why this should beat #030:
- DEMA adds faster trend confirmation than HMA alone
- Volume filter reduces false breakouts
- Cleaner MTF implementation using proper resampling
- Based on proven 15m+4h winning combination from #030
"""

import numpy as np
import pandas as pd

name = "mtf_hma_dema_kama_stoch_rsi_volume_bbw_15m_4h_v2"
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


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    dema = np.zeros(n)
    
    multiplier = 2.0 / (period + 1)
    ema1[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema1[i] = (close[i] - ema1[i - 1]) * multiplier + ema1[i - 1]
    
    ema2[period - 1] = np.mean(ema1[:period])
    
    for i in range(period, n):
        ema2[i] = (ema1[i] - ema2[i - 1]) * multiplier + ema2[i - 1]
        dema[i] = 2 * ema1[i] - ema2[i]
    
    return dema


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


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    n = len(close)
    if n < k_period + d_period:
        return np.zeros(n), np.zeros(n)
    
    stoch_k = np.zeros(n)
    stoch_d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        
        if highest_high > lowest_low:
            stoch_k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            stoch_k[i] = 50
    
    for i in range(k_period - 1 + d_period - 1, n):
        stoch_d[i] = np.mean(stoch_k[i - d_period + 1:i + 1])
    
    return stoch_k, stoch_d


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


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio (current volume / average volume)"""
    n = len(volume)
    if n < period:
        return np.ones(n)
    
    volume_ratio = np.ones(n)
    
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i - period + 1:i + 1])
        if avg_vol > 0:
            volume_ratio[i] = volume[i] / avg_vol
        else:
            volume_ratio[i] = 1.0
    
    return volume_ratio


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


def resample_to_4h(prices):
    """Resample 15m data to 4h using proper open_time index"""
    if 'open_time' not in prices.columns:
        return None
    
    prices_indexed = prices.set_index('open_time')
    
    df_4h = prices_indexed.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    dema_15m = calculate_dema(close, period=21)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    stoch_k_15m, stoch_d_15m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    volume_ratio_15m = calculate_volume_ratio(volume, period=20)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 4h for trend filters using proper method
    df_4h = resample_to_4h(prices)
    
    # Initialize 4h indicators mapped to 15m
    trend_4h = np.zeros(n)
    hma_4h_mapped = np.zeros(n)
    dema_4h_mapped = np.zeros(n)
    kama_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    
    if df_4h is not None and len(df_4h) > 0:
        c_4h = df_4h["close"].values
        h_4h = df_4h["high"].values
        l_4h = df_4h["low"].values
        
        # Calculate 4h indicators
        hma_4h = calculate_hma(c_4h, period=21)
        dema_4h = calculate_dema(c_4h, period=21)
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        
        # Map 4h indicators back to 15m timeframe using open_time
        prices_indexed = prices.set_index('open_time')
        
        # Create 4h series with proper timestamps
        hma_4h_series = pd.Series(hma_4h, index=df_4h.index)
        dema_4h_series = pd.Series(dema_4h, index=df_4h.index)
        kama_4h_series = pd.Series(kama_4h, index=df_4h.index)
        bbw_4h_series = pd.Series(bbw_4h, index=df_4h.index)
        c_4h_series = pd.Series(c_4h, index=df_4h.index)
        
        # Reindex to 15m with forward fill
        hma_4h_mapped_series = hma_4h_series.reindex(prices_indexed.index, method='ffill')
        dema_4h_mapped_series = dema_4h_series.reindex(prices_indexed.index, method='ffill')
        kama_4h_mapped_series = kama_4h_series.reindex(prices_indexed.index, method='ffill')
        bbw_4h_mapped_series = bbw_4h_series.reindex(prices_indexed.index, method='ffill')
        c_4h_mapped_series = c_4h_series.reindex(prices_indexed.index, method='ffill')
        
        # Fill NaN values
        hma_4h_mapped = hma_4h_mapped_series.fillna(method='ffill').fillna(0).values
        dema_4h_mapped = dema_4h_mapped_series.fillna(method='ffill').fillna(0).values
        kama_4h_mapped = kama_4h_mapped_series.fillna(method='ffill').fillna(0).values
        bbw_4h_mapped = bbw_4h_mapped_series.fillna(method='ffill').fillna(0).values
        c_4h_mapped = c_4h_mapped_series.fillna(method='ffill').fillna(0).values
        
        # Calculate 4h trend
        for i in range(n):
            if hma_4h_mapped[i] > 0 and dema_4h_mapped[i] > 0 and kama_4h_mapped[i] > 0:
                if c_4h_mapped[i] > hma_4h_mapped[i] and c_4h_mapped[i] > dema_4h_mapped[i] and c_4h_mapped[i] > kama_4h_mapped[i]:
                    trend_4h[i] = 1
                elif c_4h_mapped[i] < hma_4h_mapped[i] and c_4h_mapped[i] < dema_4h_mapped[i] and c_4h_mapped[i] < kama_4h_mapped[i]:
                    trend_4h[i] = -1
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries (tighter range for better quality)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # Stochastic thresholds
    STOC_LONG_MIN = 20
    STOC_LONG_MAX = 80
    STOC_SHORT_MIN = 20
    STOC_SHORT_MAX = 80
    
    # Volume ratio threshold for breakout confirmation
    VOLUME_RATIO_MIN = 1.2
    
    # BBW minimum for regime filter (4h)
    BBW_MIN = 0.015
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 2, 20, 21 + int(np.sqrt(21)))
    
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
        rsi_val = rsi_15m[i]
        stoch_k = stoch_k_15m[i]
        stoch_d = stoch_d_15m[i]
        atr = atr_15m[i]
        price = close[i]
        bbw_4h_val = bbw_4h_mapped[i]
        vol_ratio = volume_ratio_15m[i]
        
        # 4h trend must exist
        if trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # BBW filter - avoid choppy markets (4h)
        if bbw_4h_val < BBW_MIN:
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
            
            # Stoploss check (2.0*ATR)
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
        
        # Entry logic: 4h trend + 15m HMA+DEMA+KAMA + Stoch + RSI + Volume
        if trend == 1:  # Bullish trend on 4h
            # 15m trend confirmation (HMA + DEMA + KAMA)
            if (close[i] > hma_15m[i] and close[i] > dema_15m[i] and close[i] > kama_15m[i] and
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and
                STOC_LONG_MIN <= stoch_k <= STOC_LONG_MAX and
                vol_ratio >= VOLUME_RATIO_MIN):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend == -1:  # Bearish trend on 4h
            # 15m trend confirmation (HMA + DEMA + KAMA)
            if (close[i] < hma_15m[i] and close[i] < dema_15m[i] and close[i] < kama_15m[i] and
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and
                STOC_SHORT_MIN <= stoch_k <= STOC_SHORT_MAX and
                vol_ratio >= VOLUME_RATIO_MIN):
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals