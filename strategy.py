#!/usr/bin/env python3
"""
EXPERIMENT #064 - REGIME_ADAPTIVE_ENSEMBLE_MTF_HMA_MACD_RSI_ZSCORE_15M_4H_V1
==================================================================================================
Hypothesis: Ensemble of 3 independent signal types (trend + momentum + mean reversion) with
regime-adaptive sizing will outperform single-indicator approaches. Key innovations:

1. MTF using mtf_data helper (CRITICAL - 46 strategies failed audit without proper MTF)
2. 3-signal ensemble: HMA trend (4h), MACD momentum (15m), RSI+Z-score mean reversion (15m)
3. Regime detection via BBW percentile - scale position size by volatility regime
4. Voting system: need 2/3 signals agreeing for entry
5. Conservative sizing: max 0.35, discrete levels (0.0, ±0.20, ±0.35)
6. ATR stoploss at 2.5*ATR, take profit at 2R with trailing

Why this should beat current best (Sharpe=3.653):
- Ensemble reduces false signals from any single indicator
- Regime-adaptive sizing captures more in trending markets, protects in choppy
- Proper MTF alignment avoids look-ahead bias that killed 46 previous strategies
- Based on lessons from #055, #060, #062 (keeping strategies with Sharpe > 0.1)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_ensemble_mtf_hma_macd_rsi_zscore_15m_4h_v1"
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
    
    close = np.asarray(close, dtype=np.float64)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma = np.zeros((2, n))
    
    for j, p in enumerate([half_period, period]):
        for i in range(p - 1, n):
            weights = np.arange(1, p + 1)
            wma[j, i] = np.sum(close[i - p + 1:i + 1] * weights) / np.sum(weights)
    
    hma = np.zeros(n)
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma[0, start_idx:i + 1] - wma[1, start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    valid_start = slow + signal - 1
    signal_line[valid_start] = np.mean(macd_line[slow:valid_start + 1])
    
    for i in range(valid_start + 1, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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
        percentile[i] = np.sum(window <= current) / len(window)
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # 4h trend filter using mtf_data helper (CRITICAL - proper MTF alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        hma_4h = calculate_hma(close_4h, period=48)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.ones(n)
        bbw_4h_aligned = np.zeros(n)
    
    # Generate signals with ensemble voting
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.10
    
    # Thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    ZSCORE_MAX = 1.8
    MACD_HIST_MIN = 0
    
    # Regime thresholds
    BBW_PCT_TRENDING = 0.6  # Above this = trending regime (larger positions)
    BBW_PCT_CHOPPY = 0.3    # Below this = choppy regime (smaller positions)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 100, 48, 26 + 9, 20, 14)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        trend_4h = 0
        if hma_4h_aligned[i] > 0:
            if close[i] > hma_4h_aligned[i]:
                trend_4h = 1
            elif close[i] < hma_4h_aligned[i]:
                trend_4h = -1
        
        # Signal 1: HMA trend alignment (15m vs 4h)
        hma_15m = calculate_hma(close, period=21)
        hma_trend_15m = 0
        if hma_15m[i] > 0:
            if close[i] > hma_15m[i]:
                hma_trend_15m = 1
            elif close[i] < hma_15m[i]:
                hma_trend_15m = -1
        
        signal_hma = 0
        if trend_4h == 1 and hma_trend_15m == 1:
            signal_hma = 1
        elif trend_4h == -1 and hma_trend_15m == -1:
            signal_hma = -1
        
        # Signal 2: MACD momentum
        signal_macd = 0
        if macd_hist_15m[i] > MACD_HIST_MIN and macd_15m[i] > macd_signal_15m[i]:
            signal_macd = 1
        elif macd_hist_15m[i] < -MACD_HIST_MIN and macd_15m[i] < macd_signal_15m[i]:
            signal_macd = -1
        
        # Signal 3: RSI + Z-score mean reversion
        signal_rsi_z = 0
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        
        if trend_4h == 1 and (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX) and abs(zscore_val) < ZSCORE_MAX:
            signal_rsi_z = 1
        elif trend_4h == -1 and (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX) and abs(zscore_val) < ZSCORE_MAX:
            signal_rsi_z = -1
        
        # Ensemble voting: need 2/3 signals agreeing
        vote_sum = signal_hma + signal_macd + signal_rsi_z
        vote_count = sum([1 for s in [signal_hma, signal_macd, signal_rsi_z] if s != 0])
        
        # Regime-adaptive position sizing
        bbw_pct = bbw_pct_15m[i]
        if bbw_pct >= BBW_PCT_TRENDING:
            size_mult = 1.0  # Trending regime - full size
        elif bbw_pct <= BBW_PCT_CHOPPY:
            size_mult = 0.5  # Choppy regime - half size
        else:
            size_mult = 0.75  # Normal regime
        
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
            
            atr = atr_15m[i]
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = SIZE_HALF * size_mult
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -SIZE_HALF * size_mult
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered and ensemble still agrees
            if vote_sum >= 2 and prev_side == 1:
                signals[i] = SIZE_FULL * size_mult
                position_side[i] = 1
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            elif vote_sum <= -2 and prev_side == -1:
                signals[i] = -SIZE_FULL * size_mult
                position_side[i] = -1
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Exit if ensemble no longer agrees
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # Entry logic: need 2/3 signals agreeing
        if vote_count >= 2:
            if vote_sum >= 2:  # Bullish
                signals[i] = SIZE_FULL * size_mult
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            elif vote_sum <= -2:  # Bearish
                signals[i] = -SIZE_FULL * size_mult
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals