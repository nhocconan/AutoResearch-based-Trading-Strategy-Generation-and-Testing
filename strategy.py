#!/usr/bin/env python3
"""
EXPERIMENT #070 - Regime Adaptive Ensemble with MTF Voting (15m + 1h + 4h)
==================================================================================================
Hypothesis: Combine regime-adaptive position sizing with multi-timeframe signal voting.
Key innovation: Bollinger Band Width percentile determines regime (trend vs mean-revert),
then weights 3 strategies differently based on regime. More signal agreement = larger position.

Why this should beat current best (Sharpe=3.653):
- Regime detection adapts to market conditions (trend follow in low vol, MR in high vol)
- 3-strategy ensemble reduces false signals (Supertrend + RSI + MACD)
- Adaptive sizing: 0.20 (1 signal) → 0.30 (2 signals) → 0.35 (3 signals agree)
- Proper MTF using mtf_data helper (no manual resampling bugs)
- 15m entries with 1h trend + 4h regime filters

Position sizing rules (CRITICAL):
- MAX signal: 0.35 (never 1.0!)
- Discrete levels: 0.0, ±0.20, ±0.30, ±0.35
- Stoploss: 2.0*ATR trailing
- Take profit: reduce to half at 2R, trail at 1R
"""

import numpy as np
import pandas as pd

name = "regime_ensemble_mtf_voting_adaptive_15m_1h_4h_v1"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    # EMA Fast
    multiplier_fast = 2 / (fast + 1)
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = (close[i] - ema_fast[i - 1]) * multiplier_fast + ema_fast[i - 1]
    
    # EMA Slow
    multiplier_slow = 2 / (slow + 1)
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = (close[i] - ema_slow[i - 1]) * multiplier_slow + ema_slow[i - 1]
    
    # MACD Line
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal Line
    multiplier_signal = 2 / (signal + 1)
    signal_line[slow + signal - 2] = np.mean(macd_line[slow - 1:slow + signal - 1])
    for i in range(slow + signal - 1, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier_signal + signal_line[i - 1]
    
    # Histogram
    for i in range(slow + signal - 2, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


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
        percentile[i] = np.sum(window <= current) / len(window)
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    from mtf_data import get_htf_data, align_htf_to_ltf
    
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === 15m indicators (entry timeframe) ===
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_15m, macd_sig_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # === 1h indicators (trend filter) using mtf_data helper ===
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        rsi_1h = calculate_rsi(c_1h, period=14)
        supertrend_1h, st_dir_1h = calculate_supertrend(h_1h, l_1h, c_1h, period=10, multiplier=3.0)
        macd_1h, _, macd_hist_1h = calculate_macd(c_1h, fast=12, slow=26, signal=9)
        
        # Align 1h indicators to 15m timeframe
        rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
        st_dir_1h_aligned = align_htf_to_ltf(prices, df_1h, st_dir_1h)
        macd_hist_1h_aligned = align_htf_to_ltf(prices, df_1h, macd_hist_1h)
    except Exception:
        # Fallback if mtf_data fails
        rsi_1h_aligned = np.zeros(n)
        st_dir_1h_aligned = np.ones(n)
        macd_hist_1h_aligned = np.zeros(n)
    
    # === 4h indicators (regime detection) using mtf_data helper ===
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
        
        # Align 4h regime to 15m timeframe
        bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
    except Exception:
        # Fallback if mtf_data fails
        bbw_pct_4h_aligned = np.zeros(n) + 0.5  # Neutral regime
    
    # === Position sizing constants (CRITICAL for drawdown control) ===
    SIZE_SINGLE = 0.20   # 1 signal agrees
    SIZE_DOUBLE = 0.30   # 2 signals agree
    SIZE_TRIPLE = 0.35   # 3 signals agree (max!)
    
    # === Regime thresholds ===
    REGIME_TREND_LOW = 0.30   # BBW percentile < 30% = low vol = trend follow
    REGIME_MR_HIGH = 0.70     # BBW percentile > 70% = high vol = mean revert
    
    # === Signal thresholds ===
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    MACD_HIST_MIN = 0
    
    # === Stoploss/Take profit ===
    ATR_STOP_MULT = 2.0
    
    # === Initialize arrays ===
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    first_valid = max(200, 100)  # Need enough data for indicators
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi = rsi_15m[i]
        macd_hist = macd_hist_15m[i]
        st_dir = st_dir_15m[i]
        
        # 1h trend filters
        rsi_1h = rsi_1h_aligned[i]
        st_dir_1h = st_dir_1h_aligned[i]
        macd_hist_1h = macd_hist_1h_aligned[i]
        
        # 4h regime
        bbw_pct = bbw_pct_4h_aligned[i]
        
        # === Determine regime ===
        if bbw_pct < REGIME_TREND_LOW:
            regime = 'trend'  # Low volatility - follow trend
        elif bbw_pct > REGIME_MR_HIGH:
            regime = 'mean_revert'  # High volatility - mean reversion
        else:
            regime = 'neutral'
        
        # === Generate 3 strategy signals ===
        signal_count = 0
        signal_direction = 0
        
        # Strategy 1: Supertrend trend following
        if st_dir == 1 and st_dir_1h == 1:
            signal_count += 1
            signal_direction += 1
        elif st_dir == -1 and st_dir_1h == -1:
            signal_count += 1
            signal_direction -= 1
        
        # Strategy 2: RSI mean reversion (regime-dependent)
        if regime == 'mean_revert' or regime == 'neutral':
            if rsi < RSI_LONG_MAX and st_dir_1h >= 0:
                signal_count += 1
                signal_direction += 1
            elif rsi > RSI_SHORT_MIN and st_dir_1h <= 0:
                signal_count += 1
                signal_direction -= 1
        
        # Strategy 3: MACD momentum (regime-dependent)
        if regime == 'trend' or regime == 'neutral':
            if macd_hist > MACD_HIST_MIN and macd_hist_1h > 0:
                signal_count += 1
                signal_direction += 1
            elif macd_hist < -MACD_HIST_MIN and macd_hist_1h < 0:
                signal_count += 1
                signal_direction -= 1
        
        # === Check existing position for stoploss/take profit ===
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
                    signals[i] = np.sign(signals[i - 1]) * SIZE_SINGLE
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
                    signals[i] = -np.sign(signals[i - 1]) * SIZE_SINGLE
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
            
            # Hold position if no exit triggered and signals still agree
            if signal_direction != 0 and np.sign(signal_direction) == prev_side:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Exit if signals no longer agree
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # === New entry logic ===
        if signal_count >= 1 and signal_direction != 0:
            # Adaptive position sizing based on signal agreement
            if signal_count >= 3:
                position_size = SIZE_TRIPLE
            elif signal_count >= 2:
                position_size = SIZE_DOUBLE
            else:
                position_size = SIZE_SINGLE
            
            if signal_direction > 0:
                signals[i] = position_size
                position_side[i] = 1
            else:
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