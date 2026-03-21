#!/usr/bin/env python3
"""
EXPERIMENT #064 - ENSEMBLE_REGIME_CONFIDENCE_HMA_ST_RSI_ZSCORE_BBW_15M_4H_V1
==================================================================================================
Hypothesis: Combine 3 independent signal generators with regime-based weighting.
- Signal 1: Trend (HMA + Supertrend agreement on 4h)
- Signal 2: Momentum (RSI + MACD histogram on 15m)
- Signal 3: Mean Reversion (Z-score + Bollinger position on 15m)

Regime Detection:
- BBW percentile (rolling 100 bars) → low vol = trend follow, high vol = mean revert
- ADX (4h) → trend strength filter
- Volatility clustering (ATR ratio) → reduce size in high vol

Confidence Scaling:
- 1 signal agrees: 0.20 position
- 2 signals agree: 0.28 position
- 3 signals agree: 0.35 position

Key improvements over #040:
- True ensemble voting (3 independent signals)
- Regime-adaptive (trend vs mean-reversion mode)
- Confidence-based position sizing
- Hysteresis on signal changes (reduce churn)
- Cleaner state management

Timeframe: 15m entries + 4h trend filter (16 bars per 4h)
Position Size: 0.20-0.35 (discrete levels)
Stoploss: 2.0*ATR trailing
Take Profit: 2R → half position, trail at 1R
"""

import numpy as np
import pandas as pd

name = "ensemble_regime_confidence_hma_st_rsi_zscore_bbw_15m_4h_v1"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    for i in range(slow + signal, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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
    
    return adx


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


def calculate_bbw_percentile(bbw, period=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(period - 1, n):
        window = bbw[i - period + 1:i + 1]
        current = bbw[i]
        percentile[i] = np.sum(window <= current) / period
    
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
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    macd_line_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    upper_15m, middle_15m, lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, period=100)
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    bars_per_4h = 16
    n_4h = (n // bars_per_4h)
    
    # Create 4h arrays by downsampling
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = start_idx + bars_per_4h
        c_4h[i] = close[end_idx - 1]
        h_4h[i] = np.max(high[start_idx:end_idx])
        l_4h[i] = np.min(low[start_idx:end_idx])
    
    # 4h indicators for trend
    hma_4h = calculate_hma(c_4h, period=21)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            st_trend_4h[i] = st_direction_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
    
    # Generate signals with ensemble voting
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on confidence
    SIZE_LOW = 0.20    # 1 signal agrees
    SIZE_MED = 0.28    # 2 signals agree
    SIZE_HIGH = 0.35   # 3 signals agree
    
    # Thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    ZSCORE_MAX = 2.0
    ADX_MIN = 20
    BBW_PCT_LOW = 0.30   # Low volatility regime
    BBW_PCT_HIGH = 0.70  # High volatility regime
    MACD_HIST_MIN = 0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Hysteresis threshold (signal must change by this much to flip)
    HYSTERESIS = 0.15
    
    first_valid = max(200, 40 * bars_per_4h, 100, 28)
    
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
        
        # Regime detection
        bbw_pct = bbw_pct_15m[i]
        adx_val = adx_4h_mapped[i]
        
        # Skip if ADX too low (no trend)
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Determine regime
        if bbw_pct < BBW_PCT_LOW:
            regime = "trend"      # Low vol = trend following
        elif bbw_pct > BBW_PCT_HIGH:
            regime = "mean_rev"   # High vol = mean reversion
        else:
            regime = "neutral"    # Medium vol = mixed
        
        # Signal 1: Trend signal (4h HMA + Supertrend agreement)
        trend_signal = 0
        if trend_4h[i] == 1 and st_trend_4h[i] == 1:
            trend_signal = 1
        elif trend_4h[i] == -1 and st_trend_4h[i] == -1:
            trend_signal = -1
        
        # Signal 2: Momentum signal (15m RSI + MACD)
        momentum_signal = 0
        rsi_val = rsi_15m[i]
        macd_hist = macd_hist_15m[i]
        
        if regime == "trend":
            # In trend regime: RSI pullback + MACD confirmation
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and macd_hist > MACD_HIST_MIN:
                momentum_signal = 1
            elif RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and macd_hist < -MACD_HIST_MIN:
                momentum_signal = -1
        else:
            # In mean reversion regime: RSI extremes
            if rsi_val < 35 and macd_hist > 0:
                momentum_signal = 1
            elif rsi_val > 65 and macd_hist < 0:
                momentum_signal = -1
        
        # Signal 3: Mean reversion signal (15m Z-score + BB position)
        mr_signal = 0
        zscore_val = zscore_15m[i]
        bb_position = (close[i] - lower_15m[i]) / (upper_15m[i] - lower_15m[i] + 1e-10)
        
        if regime == "mean_rev" or regime == "neutral":
            if zscore_val < -1.5 and bb_position < 0.3:
                mr_signal = 1
            elif zscore_val > 1.5 and bb_position > 0.7:
                mr_signal = -1
        else:
            # In trend regime: only extreme mean reversion
            if zscore_val < -2.0 and bb_position < 0.2:
                mr_signal = 1
            elif zscore_val > 2.0 and bb_position > 0.8:
                mr_signal = -1
        
        # Ensemble voting with regime weighting
        if regime == "trend":
            # Weight trend signal higher
            votes = trend_signal * 2 + momentum_signal + mr_signal
        elif regime == "mean_rev":
            # Weight mean reversion signal higher
            votes = trend_signal + momentum_signal + mr_signal * 2
        else:
            # Equal weighting
            votes = trend_signal + momentum_signal + mr_signal
        
        # Calculate confidence and position size
        vote_count = abs(trend_signal) + abs(momentum_signal) + abs(mr_signal)
        vote_sum = votes
        
        # Determine target signal based on vote count and direction
        target_signal = 0.0
        if vote_sum > 0 and vote_count >= 1:
            if vote_count == 1:
                target_signal = SIZE_LOW * np.sign(vote_sum)
            elif vote_count == 2:
                target_signal = SIZE_MED * np.sign(vote_sum)
            else:
                target_signal = SIZE_HIGH * np.sign(vote_sum)
        
        # Apply hysteresis (don't flip unless signal changes significantly)
        prev_signal = signals[i - 1] if i > 0 else 0.0
        if abs(target_signal - prev_signal) < HYSTERESIS and prev_signal != 0:
            target_signal = prev_signal
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            atr = atr_15m[i]
            
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
                    signals[i] = prev_side * SIZE_LOW
                    position_side[i] = prev_side
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
                    signals[i] = prev_side * SIZE_LOW
                    position_side[i] = prev_side
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
        
        # Apply target signal if no existing position
        signals[i] = target_signal
        if target_signal != 0:
            position_side[i] = np.sign(target_signal)
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        else:
            position_side[i] = 0
    
    return signals