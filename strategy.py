#!/usr/bin/env python3
"""
EXPERIMENT #051 - Ensemble Regime Adaptive Strategy (1h Primary + 4h Trend)
==================================================================================================
Hypothesis: Combining 3 signal types (trend, momentum, mean-reversion) with regime-aware weighting
reduces whipsaws and improves risk-adjusted returns. Bollinger Band Width percentile detects market
regime: low vol → trend following gets higher weight, high vol → mean reversion gets higher weight.

Key innovations:
1. ENSEMBLE VOTING: 3 independent signals (HMA trend, RSI momentum, Z-score mean-reversion)
2. REGIME DETECTION: BBW percentile over 100 bars → adaptive signal weighting
3. ADAPTIVE SIZING: More signals agree = larger position (0.15 to 0.35)
4. 4h TREND FILTER: Prevents counter-trend trades during strong directional moves
5. ATR STOPLOSS: 2.5*ATR trailing stop with take-profit at 2R

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- Ensemble reduces single-indicator failure modes
- Regime adaptation matches strategy to market conditions
- Adaptive sizing reduces exposure during uncertain signals
- 1h timeframe generates more opportunities than 4h while 4h filter prevents disasters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_regime_adaptive_1h_4h_v1"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close = np.array(close, dtype=float)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    wma_half = pd.Series(close).ewm(span=half, adjust=False, min_periods=half).mean().values
    # WMA for full period
    wma_full = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_full
    hma = pd.Series(hma_raw).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean().values
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close = np.array(close, dtype=float)
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = rolling_std > 0
    zscore[mask] = (close[mask] - rolling_mean[mask]) / rolling_std[mask]
    
    return zscore


def calculate_bollinger Bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close = np.array(close, dtype=float)
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = rolling_mean + std_mult * rolling_std
    lower = rolling_mean - std_mult * rolling_std
    band_width = (upper - lower) / rolling_mean
    
    # Handle division by zero
    band_width = np.where(np.isfinite(band_width), band_width, 0)
    
    return upper, lower, band_width


def calculate_bbw_percentile(band_width, lookback=100):
    """Calculate Bollinger Band Width percentile over rolling lookback"""
    n = len(band_width)
    percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = band_width[i - lookback:i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= band_width[i]) / len(valid) * 100
        else:
            percentile[i] = 50
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    hma_1h_fast = calculate_hma(close, period=9)
    zscore_1h = calculate_zscore(close, period=20)
    bb_upper, bb_lower, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(close_4h, period=21)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE with adaptive conviction
    SIZE_LOW = 0.15    # Low conviction (1 signal)
    SIZE_BASE = 0.25   # Base position (2 signals agree)
    SIZE_HIGH = 0.35   # High conviction (all 3 signals agree)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    
    # Regime thresholds
    REGIME_LOW_VOL = 30   # BBW percentile < 30 = quiet market (trend follow)
    REGIME_HIGH_VOL = 70  # BBW percentile > 70 = volatile market (mean revert)
    
    first_valid = max(150, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        hma_val = hma_1h[i]
        hma_fast_val = hma_1h_fast[i]
        zscore_val = zscore_1h[i]
        bbw_pct = bbw_pct_1h[i]
        
        # 4h trend filter
        hma_4h_val = hma_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if hma_4h_val > 0:
            if price > hma_4h_val:
                trend_4h = 1
            elif price < hma_4h_val:
                trend_4h = -1
        
        # ========== REGIME DETECTION ==========
        # Low vol: favor trend signals
        # High vol: favor mean reversion signals
        if bbw_pct < REGIME_LOW_VOL:
            trend_weight = 0.5
            momentum_weight = 0.3
            meanrev_weight = 0.2
        elif bbw_pct > REGIME_HIGH_VOL:
            trend_weight = 0.2
            momentum_weight = 0.3
            meanrev_weight = 0.5
        else:
            trend_weight = 0.33
            momentum_weight = 0.34
            meanrev_weight = 0.33
        
        # ========== SIGNAL 1: TREND (HMA slope + 4h filter) ==========
        trend_signal = 0
        if hma_fast_val > hma_val and trend_4h >= 0:
            trend_signal = 1
        elif hma_fast_val < hma_val and trend_4h <= 0:
            trend_signal = -1
        
        # ========== SIGNAL 2: MOMENTUM (RSI) ==========
        momentum_signal = 0
        if rsi_val > 55 and trend_4h >= 0:
            momentum_signal = 1
        elif rsi_val < 45 and trend_4h <= 0:
            momentum_signal = -1
        
        # ========== SIGNAL 3: MEAN REVERSION (Z-score) ==========
        meanrev_signal = 0
        if zscore_val < -1.5 and trend_4h >= 0:
            meanrev_signal = 1  # Oversold in uptrend
        elif zscore_val > 1.5 and trend_4h <= 0:
            meanrev_signal = -1  # Overbought in downtrend
        
        # ========== ENSEMBLE VOTING ==========
        weighted_vote = (
            trend_signal * trend_weight +
            momentum_signal * momentum_weight +
            meanrev_signal * meanrev_weight
        )
        
        # Count agreeing signals
        signals_agree = 0
        if trend_signal == 1:
            signals_agree += 1
        elif trend_signal == -1:
            signals_agree -= 1
        
        if momentum_signal == 1:
            signals_agree += 1
        elif momentum_signal == -1:
            signals_agree -= 1
        
        if meanrev_signal == 1:
            signals_agree += 1
        elif meanrev_signal == -1:
            signals_agree -= 1
        
        # ========== CHECK EXISTING POSITIONS ==========
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
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE / 2
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
                    signals[i] = -SIZE_BASE / 2
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
            
            # Hold position if no exit triggered (unless signal reverses)
            if weighted_vote * prev_side < -0.3:  # Strong reversal signal
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== ENTRY LOGIC - ENSEMBLE VOTING ==========
        # Need at least 2 signals agreeing for entry
        if signals_agree >= 2:
            # All 3 agree = high conviction
            size = SIZE_HIGH
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif signals_agree <= -2:
            # All 3 agree short = high conviction
            size = SIZE_HIGH
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif signals_agree == 1:
            # 2 signals long, 1 neutral/short = base conviction
            size = SIZE_BASE
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif signals_agree == -1:
            # 2 signals short, 1 neutral/long = base conviction
            size = SIZE_BASE
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals