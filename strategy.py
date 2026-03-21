#!/usr/bin/env python3
"""
EXPERIMENT #093 - ENSEMBLE_VOTING_REGIME_MTF_PROPER_HTF_15M_1H_4H_V1
==================================================================================================
Hypothesis: Recent ensemble failures (#085, #086, #089, #090, #092) had massive drawdowns due to:
1. Manual resampling instead of mtf_data helper (causes misalignment on SOL gaps)
2. Too many signals voting (increases churn and fees)
3. No adaptive sizing based on regime confidence

Key changes from #040:
- USE mtf_data helper for ALL HTF data (get_htf_data + align_htf_to_ltf)
- Simpler 3-signal ensemble: HMA + Supertrend + RSI (not 7+ indicators)
- Regime-based sizing: 4h BBW percentile → scale position 0.20-0.35
- 15m entries + 1h trend + 4h regime (proven multi-TF structure)
- Discrete signal levels: 0.0, ±0.20, ±0.35 (reduce churn)
- Proper stoploss: 2*ATR, TP: 2R→half, trail at 1R

Why this should beat #040:
- Proper HTF alignment (no manual resampling bugs)
- Fewer signals = less churn = lower fees
- Regime-adaptive sizing = better risk control in high vol
- Based on winning pattern from #031, #034, #035 (15m + MTF filters)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_voting_regime_mtf_proper_htf_15m_1h_4h_v1"
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
        current = bbw[i]
        rank = np.sum(window <= current)
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Get 1h HTF data using mtf_data helper (PROPER alignment)
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        # 1h indicators
        hma_1h_raw = calculate_hma(close_1h, period=21)
        _, st_direction_1h_raw = calculate_supertrend(high_1h, low_1h, close_1h, period=10, multiplier=3.0)
        rsi_1h_raw = calculate_rsi(close_1h, period=14)
        
        # Align HTF indicators to LTF (auto shift(1) for completed bars)
        hma_1h = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
        st_direction_1h = align_htf_to_ltf(prices, df_1h, st_direction_1h_raw)
        rsi_1h = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    except Exception:
        # Fallback if mtf_data fails
        hma_1h = np.zeros(n)
        st_direction_1h = np.zeros(n)
        rsi_1h = np.zeros(n)
    
    # Get 4h HTF data for regime detection (PROPER alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        
        # 4h BBW for regime
        _, _, _, bbw_4h_raw = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        bbw_pct_4h_raw = calculate_bbw_percentile(bbw_4h_raw, lookback=100)
        
        # Align to 15m
        bbw_pct_4h = align_htf_to_ltf(prices, df_4h, bbw_pct_4h_raw)
    except Exception:
        bbw_pct_4h = np.zeros(n)
    
    # Generate signals with ensemble voting + regime sizing
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on regime
    SIZE_LOW_VOL = 0.35  # BBW percentile < 0.5 (trend regime)
    SIZE_HIGH_VOL = 0.20  # BBW percentile >= 0.5 (choppy regime)
    SIZE_HALF_LOW = 0.175
    SIZE_HALF_HIGH = 0.10
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
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
        
        # Get regime-based position size
        regime_pct = bbw_pct_4h[i] if bbw_pct_4h[i] > 0 else 0.5
        if regime_pct < 0.5:
            size_full = SIZE_LOW_VOL
            size_half = SIZE_HALF_LOW
        else:
            size_full = SIZE_HIGH_VOL
            size_half = SIZE_HALF_HIGH
        
        # 1h trend filters (using aligned HTF data)
        hma_trend_1h = 0
        if hma_1h[i] > 0:
            if close[i] > hma_1h[i]:
                hma_trend_1h = 1
            elif close[i] < hma_1h[i]:
                hma_trend_1h = -1
        
        st_trend_1h = st_direction_1h[i] if st_direction_1h[i] != 0 else 0
        rsi_1h_val = rsi_1h[i] if rsi_1h[i] > 0 else 50
        
        # 15m entry signals
        st_trend_15m = st_direction_15m[i]
        rsi_15m_val = rsi_15m[i]
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, close[i])
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = size_half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -size_half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if close[i] > trail_stop:
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
        
        # ENSEMBLE VOTING: Need 2/3 signals to agree
        # Signal 1: 1h HMA trend
        # Signal 2: 1h Supertrend
        # Signal 3: 15m RSI pullback in direction of trend
        
        vote_long = 0
        vote_short = 0
        
        # 1h HMA trend vote
        if hma_trend_1h == 1:
            vote_long += 1
        elif hma_trend_1h == -1:
            vote_short += 1
        
        # 1h Supertrend vote
        if st_trend_1h == 1:
            vote_long += 1
        elif st_trend_1h == -1:
            vote_short += 1
        
        # 15m RSI pullback vote (only if 1h trend is clear)
        if hma_trend_1h == 1 or st_trend_1h == 1:
            if RSI_LONG_MIN <= rsi_15m_val <= RSI_LONG_MAX:
                vote_long += 1
        elif hma_trend_1h == -1 or st_trend_1h == -1:
            if RSI_SHORT_MIN <= rsi_15m_val <= RSI_SHORT_MAX:
                vote_short += 1
        
        # Entry logic: Need 2/3 votes
        if vote_long >= 2:
            signals[i] = size_full
            position_side[i] = 1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
                
        elif vote_short >= 2:
            signals[i] = -size_full
            position_side[i] = -1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals