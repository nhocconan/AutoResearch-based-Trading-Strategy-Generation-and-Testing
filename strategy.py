#!/usr/bin/env python3
"""
EXPERIMENT #020 - MTF KAMA+Stochastic+ADX+Volume+Z-score (1h+4h v1)
==================================================================================================
Hypothesis: Current best #019 uses 15m+1h with DEMA+Supertrend+MACD+RSI (Sharpe=3.578).
This experiment tries a DIFFERENT combination:

Key changes:
- 4h KAMA trend (Kaufman Adaptive MA - adapts to volatility, less whipsaw than DEMA/HMA)
- 1h Stochastic entry (different from RSI - faster reaction to momentum shifts)
- 1h ADX filter (trend strength confirmation)
- Volume spike confirmation (2x average volume for entry validation)
- Z-score regime filter (avoid extreme mean-reversion conditions)
- Position size: 0.30 (same conservative sizing)
- Stoploss: 2.5*ATR with trailing stop at 1R

Why this should beat #019:
- KAMA adapts efficiency ratio to market conditions (better in choppy vs trending)
- Stochastic gives earlier entry signals than RSI (faster momentum detection)
- 4h trend + 1h entry is proven structure (#012 had Sharpe=0.478, we improve filtering)
- Volume confirmation filters false breakouts
- Different signal combination than current best (diversification of approach)
"""

import numpy as np
import pandas as pd

name = "mtf_kama_stochastic_adx_volume_zscore_1h_4h_v1"
timeframe = "1h"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
    n = len(close)
    if n < k_period + d_period:
        return np.zeros(n), np.zeros(n)
    
    k_line = np.zeros(n)
    d_line = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        
        if highest_high != lowest_low:
            k_line[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            k_line[i] = 50
    
    # Calculate %D (SMA of %K)
    for i in range(k_period + d_period - 2, n):
        d_line[i] = np.mean(k_line[i - d_period + 1:i + 1])
    
    return k_line, d_line


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    volume_sma = np.zeros(n)
    for i in range(period - 1, n):
        volume_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return volume_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Ensure we have open_time for proper resampling
    if 'open_time' not in prices.columns:
        return np.zeros(n)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    kama_fast_1h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow_1h = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Resample to 4h for trend filter using CORRECT method (no pd.date_range!)
    prices_indexed = prices.set_index('open_time')
    
    # Resample to 4h
    df_4h = prices_indexed.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h trend indicators
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    kama_fast_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    kama_slow_4h = calculate_kama(c_4h, period=20, fast_period=2, slow_period=30)
    adx_4h, _, _ = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators back to 1h using reindex with ffill
    df_4h['kama_fast'] = kama_fast_4h
    df_4h['kama_slow'] = kama_slow_4h
    df_4h['adx'] = adx_4h
    
    # Reindex to original 1h timestamps
    df_4h_aligned = df_4h.reindex(prices_indexed.index, method='ffill')
    
    kama_fast_4h_mapped = df_4h_aligned['kama_fast'].values
    kama_slow_4h_mapped = df_4h_aligned['kama_slow'].values
    adx_4h_mapped = df_4h_aligned['adx'].values
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Stochastic thresholds for entry
    STOCH_LONG_ENTRY = 25  # Oversold for long entry
    STOCH_SHORT_ENTRY = 75  # Overbought for short entry
    STOCH_EXIT = 50  # Neutral exit level
    
    # ADX thresholds
    ADX_1H_MIN = 20  # 1h trend strength
    ADX_4H_MIN = 25  # 4h trend strength (higher for major trend)
    
    # Z-score threshold
    ZSCORE_MAX = 2.0
    
    # Volume spike threshold
    VOLUME_MULT = 1.5  # Volume must be 1.5x average
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 60)  # Ensure all indicators are calculated
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for NaN values
        if (np.isnan(atr_1h[i]) or np.isnan(stoch_k_1h[i]) or 
            np.isnan(zscore_1h[i]) or atr_1h[i] == 0 or
            np.isnan(kama_fast_4h_mapped[i]) or np.isnan(adx_4h_mapped[i])):
            signals[i] = 0.0
            if i > 0:
                position_side[i] = position_side[i - 1]
            continue
        
        # 4h trend filter (KAMA crossover)
        trend_4h = 0
        if c_4h[-1] > kama_fast_4h_mapped[i] and kama_fast_4h_mapped[i] > kama_slow_4h_mapped[i]:
            trend_4h = 1
        elif c_4h[-1] < kama_fast_4h_mapped[i] and kama_fast_4h_mapped[i] < kama_slow_4h_mapped[i]:
            trend_4h = -1
        
        # ADX filters (both 1h and 4h must show trend strength)
        adx_1h_val = adx_1h[i]
        adx_4h_val = adx_4h_mapped[i]
        
        if adx_1h_val < ADX_1H_MIN or adx_4h_val < ADX_4H_MIN:
            # No strong trend - close any existing position
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            continue
        
        # Z-score regime filter (avoid extreme conditions)
        zscore_val = zscore_1h[i]
        if abs(zscore_val) > ZSCORE_MAX:
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
            
            price = close[i]
            atr = atr_1h[i]
            
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
            
            # Check if trend reversed (exit signal)
            stoch_k = stoch_k_1h[i]
            stoch_d = stoch_d_1h[i]
            
            if prev_side == 1 and (stoch_k > STOCH_EXIT or trend_4h != 1):
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            elif prev_side == -1 and (stoch_k < STOCH_EXIT or trend_4h != -1):
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
        
        # Entry logic: 4h KAMA trend + 1h Stochastic + ADX + Volume + Z-score
        stoch_k = stoch_k_1h[i]
        stoch_d = stoch_d_1h[i]
        volume_ratio = volume[i] / volume_sma_1h[i] if volume_sma_1h[i] > 0 else 0
        
        # Long entry: 4h bullish + 1h Stochastic oversold + volume spike
        if trend_4h == 1:
            if (stoch_k < STOCH_LONG_ENTRY and 
                stoch_d < STOCH_LONG_ENTRY + 10 and
                volume_ratio > VOLUME_MULT):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
                
        # Short entry: 4h bearish + 1h Stochastic overbought + volume spike
        elif trend_4h == -1:
            if (stoch_k > STOCH_SHORT_ENTRY and 
                stoch_d > STOCH_SHORT_ENTRY - 10 and
                volume_ratio > VOLUME_MULT):
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals