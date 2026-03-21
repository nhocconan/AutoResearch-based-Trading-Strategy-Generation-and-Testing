#!/usr/bin/env python3
"""
EXPERIMENT #013 - MTF HMA+Supertrend+Stoch+RSI+ADX+Zscore (15m+1h+4h v1)
==================================================================================================
Hypothesis: Combine 4h HMA trend (from reported Sharpe=5.4 baseline) with 1h Supertrend 
confirmation, 15m Stochastic+RSI dual entry timing (new combo), and ADX strength filter 
(untried in recent winners). Z-score prevents extreme entries.

Key differences from #012:
- Replace 4h Supertrend with 4h HMA (reported best in instructions)
- Add 1h Supertrend as secondary trend confirmation
- Replace MACD with Stochastic (14,3,3) for entry timing - NEW
- Add ADX(14) > 25 filter for trend strength - NEW
- Keep RSI + Z-score filters from proven strategies
- Position size: 0.28 (balanced between 0.25-0.30)
- Stoploss: 2.0*ATR (tighter than 2.5*ATR to reduce drawdown)

Why this should beat #004 and #012:
- 4h HMA reported as Sharpe=5.4 baseline in instructions
- Stochastic adds overbought/oversold timing MACD lacks
- ADX filter removes weak trend periods (major drawdown source)
- Based on winning formula but with better entry/exit timing
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_supertrend_stoch_rsi_adx_zscore_15m_1h_4h_v1"
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


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
    n = len(close)
    if n < k_period:
        return np.zeros(n), np.zeros(n)
    
    k_line = np.zeros(n)
    d_line = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        
        if highest_high - lowest_low > 0:
            k_line[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            k_line[i] = 50
    
    for i in range(k_period - 1 + d_period - 1, n):
        d_line[i] = np.mean(k_line[i - d_period + 1:i + 1])
    
    return k_line, d_line


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 3:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    plus_di_smooth = np.zeros(n)
    minus_di_smooth = np.zeros(n)
    tr_smooth = np.zeros(n)
    
    plus_di_smooth[period - 1] = np.sum(plus_dm[1:period])
    minus_di_smooth[period - 1] = np.sum(minus_dm[1:period])
    tr_smooth[period - 1] = np.sum(tr[1:period])
    
    for i in range(period, n):
        plus_di_smooth[i] = plus_di_smooth[i - 1] - plus_di_smooth[i - 1] / period + plus_dm[i]
        minus_di_smooth[i] = minus_di_smooth[i - 1] - minus_di_smooth[i - 1] / period + minus_dm[i]
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_di_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_di_smooth[i] / tr_smooth[i]
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    stoch_k_15m, stoch_d_15m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    adx_15m = calculate_adx(high, low, close, period=14)
    
    # Get 1h HTF data using mtf_data helper (MANDATORY - no manual resampling)
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        # 1h Supertrend for trend confirmation
        _, st_direction_1h_raw = calculate_supertrend(h_1h, l_1h, c_1h, period=10, multiplier=3.0)
        st_direction_1h = align_htf_to_ltf(prices, df_1h, st_direction_1h_raw)
        
        # 1h ADX for trend strength
        adx_1h_raw = calculate_adx(h_1h, l_1h, c_1h, period=14)
        adx_1h = align_htf_to_ltf(prices, df_1h, adx_1h_raw)
    except Exception:
        st_direction_1h = np.ones(n)
        adx_1h = np.zeros(n)
    
    # Get 4h HTF data using mtf_data helper (MANDATORY)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        
        # 4h HMA for primary trend direction
        hma_4h_raw = calculate_hma(c_4h, period=21)
        hma_4h = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
        
        # 4h HMA(48) for secondary confirmation
        hma_48_4h_raw = calculate_hma(c_4h, period=48)
        hma_48_4h = align_htf_to_ltf(prices, df_4h, hma_48_4h_raw)
    except Exception:
        hma_4h = np.zeros(n)
        hma_48_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.28  # Conservative position size
    SIZE_HALF = 0.14
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Stochastic thresholds
    STOC_LONG_MAX = 40  # Enter when %K crosses above from oversold
    STOC_SHORT_MIN = 60  # Enter when %K crosses below from overbought
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ADX threshold for trend strength
    ADX_MIN = 25
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 42, 20, 28)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Track stochastic cross for entry timing
    prev_stoch_k = np.zeros(n)
    
    for i in range(first_valid, n):
        prev_stoch_k[i] = stoch_k_15m[i - 1] if i > 0 else stoch_k_15m[i]
        
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filters (primary) - HMA alignment
        hma_4h_trend = 1 if close[i] > hma_4h[i] else (-1 if close[i] < hma_4h[i] else 0)
        hma_48_4h_trend = 1 if hma_4h[i] > hma_48_4h[i] else (-1 if hma_4h[i] < hma_48_4h[i] else 0)
        
        # 1h trend filters (secondary)
        st_trend_1h = st_direction_1h[i]
        adx_1h_strength = adx_1h[i]
        
        # 15m entry filters
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        stoch_k_val = stoch_k_15m[i]
        stoch_d_val = stoch_d_15m[i]
        adx_15m_val = adx_15m[i]
        price = close[i]
        atr = atr_15m[i]
        
        # Stochastic cross detection
        stoch_cross_long = prev_stoch_k[i] < STOC_LONG_MAX and stoch_k_val >= STOC_LONG_MAX
        stoch_cross_short = prev_stoch_k[i] > STOC_SHORT_MIN and stoch_k_val <= STOC_SHORT_MIN
        
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
        
        # Entry logic: 4h HMA trend + 1h Supertrend + ADX + 15m Stoch+RSI+Zscore
        if hma_4h_trend == 1 and hma_48_4h_trend == 1 and st_trend_1h == 1:
            # Bullish trend confirmed on 4h and 1h
            if (adx_1h_strength > ADX_MIN and
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX and
                stoch_cross_long and
                stoch_k_val > stoch_d_val):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif hma_4h_trend == -1 and hma_48_4h_trend == -1 and st_trend_1h == -1:
            # Bearish trend confirmed on 4h and 1h
            if (adx_1h_strength > ADX_MIN and
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX and
                stoch_cross_short and
                stoch_k_val < stoch_d_val):
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