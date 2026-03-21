#!/usr/bin/env python3
"""
EXPERIMENT #095 - SIMPLIFIED MTF ENSEMBLE WITH PROPER HTF + REGIME FILTER
==================================================================================================
Hypothesis: Complex ensemble voting failed in #085-#094. Need SIMPLER logic with proper mtf_data.

Key changes from #040:
- Use mtf_data helper (get_htf_data, align_htf_to_ltf) - NO manual resampling!
- Simpler ensemble: 3 signals must agree (HMA trend + Supertrend + RSI pullback)
- Regime filter: BBW percentile (trend-follow in low vol, avoid high vol chop)
- Discrete signal levels: 0.0, ±0.20, ±0.35 (reduce churning costs)
- Tighter stoploss: 1.5*ATR (better R:R than 2.0*ATR)
- Position sizing: scale by signal confidence (3/3 agreement = 0.35, 2/3 = 0.20)

Why this should work:
- Proper MTF alignment prevents look-ahead bias (46 strategies failed without this)
- Simpler logic = less overfitting (complex ensembles failed with -85% to -97% DD)
- Regime filter avoids choppy markets where trend strategies fail
- Based on current best: mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1 (Sharpe=3.653)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "simplified_mtf_ensemble_proper_htf_regime_15m_4h_v1"
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
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === 15m indicators (entry timeframe) ===
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # === 4h indicators (trend filter) - USE mtf_data helper ===
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        hma_4h_raw = calculate_hma(close_4h, period=21)
        st_4h_raw, st_direction_4h_raw = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        _, _, _, bbw_4h_raw = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        bbw_pct_4h_raw = calculate_bbw_percentile(bbw_4h_raw, lookback=100)
        
        # Align HTF indicators to LTF (auto shift for completed bars)
        hma_4h = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
        st_direction_4h = align_htf_to_ltf(prices, df_4h, st_direction_4h_raw)
        bbw_pct_4h = align_htf_to_ltf(prices, df_4h, bbw_pct_4h_raw)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h = np.zeros(n)
        st_direction_4h = np.zeros(n)
        bbw_pct_4h = np.zeros(n)
    
    # === Signal parameters ===
    SIZE_FULL = 0.35
    SIZE_HALF = 0.20
    SIZE_QUARTER = 0.10
    
    ATR_STOP_MULT = 1.5
    TP_MULT = 2.0
    
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    BBW_PCT_LOW = 0.30
    BBW_PCT_HIGH = 0.70
    
    first_valid = max(200, 100, 21, 14)
    
    # === Track position state ===
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        st_15m = st_direction_15m[i]
        bbw_pct = bbw_pct_15m[i]
        bbw_pct_4h_val = bbw_pct_4h[i]
        
        # === 4h Trend signals ===
        hma_trend_4h = 0
        if hma_4h[i] > 0:
            if price > hma_4h[i]:
                hma_trend_4h = 1
            elif price < hma_4h[i]:
                hma_trend_4h = -1
        
        st_trend_4h = st_direction_4h[i]
        
        # === Regime filter (avoid choppy high-vol markets) ===
        # Low BBW percentile = trending regime (good for trend-follow)
        # High BBW percentile = mean-reversion regime (avoid for trend strategies)
        regime_trend = bbw_pct_4h_val < BBW_PCT_HIGH
        
        # === Count signal agreement ===
        bullish_signals = 0
        bearish_signals = 0
        
        if hma_trend_4h == 1:
            bullish_signals += 1
        elif hma_trend_4h == -1:
            bearish_signals += 1
        
        if st_trend_4h == 1:
            bullish_signals += 1
        elif st_trend_4h == -1:
            bearish_signals += 1
        
        if st_15m == 1:
            bullish_signals += 1
        elif st_15m == -1:
            bearish_signals += 1
        
        # === Check existing positions ===
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
            
            # Stoploss check
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
                
                # Take profit at 2R
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
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
                
                # Take profit at 2R
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
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
            if prev_side == 1 and (bearish_signals >= 2 or not regime_trend):
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            if prev_side == -1 and (bullish_signals >= 2 or not regime_trend):
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
        
        # === Entry logic ===
        if not regime_trend:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Long entry: 4h HMA + 4h Supertrend + 15m Supertrend bullish
        if bullish_signals >= 2 and hma_trend_4h == 1:
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                # Scale position by signal confidence
                if bullish_signals == 3:
                    signals[i] = SIZE_FULL
                else:
                    signals[i] = SIZE_HALF
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        # Short entry: 4h HMA + 4h Supertrend + 15m Supertrend bearish
        elif bearish_signals >= 2 and hma_trend_4h == -1:
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                if bearish_signals == 3:
                    signals[i] = -SIZE_FULL
                else:
                    signals[i] = -SIZE_HALF
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals