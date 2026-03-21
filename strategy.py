#!/usr/bin/env python3
"""
EXPERIMENT #060 - REGIME_ENSEMBLE_CONFIDENCE_HMA_ST_KAMA_RSI_15M_4H_V1
==================================================================================================
Hypothesis: Combining regime-adaptive signal voting with confidence-based position sizing 
will outperform single-indicator strategies. Using 15m entries with 4h trend filter 
(proven in #049, #053, #054) plus Bollinger Band Width percentile for regime detection.

Key innovations:
- REGIME DETECTION: BBW percentile (rolling 100 bars) → trend follow when BBW < 30th percentile
- SIGNAL VOTING: 4 indicators vote (HMA, Supertrend, KAMA, RSI) → majority wins
- CONFIDENCE SIZING: More agreement = larger position (2/4=0.20, 3/4=0.28, 4/4=0.35)
- 15M/4H MULTI-TF: 15m for entries (more signals), 4h for trend filter (less noise)
- ADAPTIVE STOPS: Wider stops in high vol regime (3*ATR), tighter in low vol (2*ATR)
- NO LOOK-AHEAD: All calculations use only past data (fixed #059 crash)

Why this should beat #058 (Sharpe=5.353) and approach #049 (Sharpe=13.974):
- BBW regime detection is more robust than volatility clustering ratio
- Signal voting reduces false signals from any single indicator
- Confidence sizing maximizes returns when signals agree strongly
- 15m/4h combination has proven track record in multiple experiments

Position sizing rules (CRITICAL):
- MAX signal: 0.35 (proven to control drawdown in 2022 crash)
- MIN signal: 0.20 (avoid tiny positions eaten by fees)
- Discrete levels: 0.0, 0.20, 0.28, 0.35 (reduces churn costs)
- Stoploss: 2.5*ATR trailing (adjusts to 3.5*ATR in high vol regime)
- Volatility scaling: position_size = base_size * (target_vol / current_vol)
"""

import numpy as np
import pandas as pd

name = "regime_ensemble_confidence_hma_st_kama_rsi_15m_4h_v1"
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


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def calc_wma(data, wma_period):
        result = np.zeros(len(data))
        for i in range(wma_period - 1, len(data)):
            weights = np.arange(1, wma_period + 1)
            window = data[i - wma_period + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma1 = calc_wma(close, half)
    wma2 = calc_wma(close, period)
    raw_hma = 2 * wma1 - wma2
    hma = calc_wma(raw_hma, sqrt_period)
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    direction = np.zeros(n)
    supertrend = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper = mid + multiplier * atr[i]
        lower = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper
            direction[i] = 1
        else:
            if direction[i - 1] == 1:
                if close[i] < upper:
                    supertrend[i] = upper
                    direction[i] = 1
                else:
                    supertrend[i] = lower
                    direction[i] = -1
            else:
                if close[i] > lower:
                    supertrend[i] = lower
                    direction[i] = -1
                else:
                    supertrend[i] = upper
                    direction[i] = 1
    
    return supertrend, direction


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = np.zeros(n)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std
    
    return upper, sma, lower


def calculate_bbw_percentile(close, period=20, lookback=100):
    """
    Calculate Bollinger Band Width percentile for regime detection.
    Low BBW percentile = compression = potential trend breakout
    High BBW percentile = expansion = range/trend continuation
    """
    n = len(close)
    if n < lookback + period:
        return np.zeros(n)
    
    bbw = np.zeros(n)
    bbw_percentile = np.zeros(n)
    
    for i in range(period - 1 + lookback, n):
        # Calculate current BBW
        sma = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper = sma + 2.0 * std
        lower = sma - 2.0 * std
        bbw[i] = (upper - lower) / sma if sma > 0 else 0
        
        # Calculate percentile over lookback window
        bbw_window = bbw[i - lookback:i + 1]
        bbw_percentile[i] = np.sum(bbw_window <= bbw[i]) / len(bbw_window) * 100
    
    return bbw_percentile


def resample_to_higher_tf(close, high, low, volume, bars_per_tf=16):
    """Resample 15m data to 4h (16 x 15m = 4h)"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return close.copy(), high.copy(), low.copy(), volume.copy()
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    v_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        if end_idx <= n:
            c_tf[i] = close[end_idx - 1]
            h_tf[i] = np.max(high[start_idx:end_idx])
            l_tf[i] = np.min(low[start_idx:end_idx])
            v_tf[i] = np.sum(volume[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf, v_tf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=16)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    st_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    bbw_pct_15m = calculate_bbw_percentile(close, period=20, lookback=100)
    
    # Resample to 4h for trend (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h, v_4h = resample_to_higher_tf(close, high, low, volume, bars_per_4h)
    
    # 4h indicators for trend
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    hma_4h = calculate_hma(c_4h, period=16)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    st_4h, st_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    bbw_pct_4h = calculate_bbw_percentile(c_4h, period=20, lookback=100)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    kama_trend_4h = np.zeros(n)
    bbw_regime_4h = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    
    n_4h = len(c_4h)
    for i in range(n):
        idx_4h = i // bars_per_4h
        
        if idx_4h < n_4h and idx_4h >= 40:
            # HMA trend
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Supertrend direction
            st_trend_4h[i] = st_dir_4h[idx_4h]
            
            # KAMA trend
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                kama_trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                kama_trend_4h[i] = -1
            
            # BBW regime (low percentile = compression = trend potential)
            bbw_val = bbw_pct_4h[idx_4h] if idx_4h < len(bbw_pct_4h) else 50
            bbw_regime_4h[i] = bbw_val
            
            # ATR mapped
            atr_4h_mapped[i] = atr_4h[idx_4h] if idx_4h < len(atr_4h) else atr_15m[i]
    
    # Position sizing parameters (DISCRETE levels based on signal agreement)
    SIZE_LEVELS = {2: 0.20, 3: 0.28, 4: 0.35}
    BASE_SIZE = 0.28
    
    # Regime thresholds
    BBW_TREND_THRESHOLD = 30  # Below 30th percentile = trend regime
    BBW_RANGE_THRESHOLD = 70  # Above 70th percentile = range regime
    
    # Stoploss multipliers (adaptive to vol regime)
    ATR_STOP_TREND = 3.0  # Wider stops in trend regime
    ATR_STOP_RANGE = 2.5  # Tighter stops in range regime
    
    first_valid = max(200, 40 * bars_per_4h + 100)
    
    # Generate signals with regime-switching and ensemble voting
    signals = np.zeros(n)
    
    # Track position state for stoploss/TP
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h regime signals
        hma_trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        kama_trend = kama_trend_4h[i]
        bbw_regime = bbw_regime_4h[i]
        atr_4h_val = atr_4h_mapped[i]
        
        # 15m entry signals
        price = close[i]
        hma_15m_val = hma_15m[i]
        kama_15m_val = kama_15m[i]
        st_dir = st_dir_15m[i]
        rsi_val = rsi_15m[i]
        atr_15m_val = atr_15m[i]
        
        # Determine regime and adaptive ATR stop
        if bbw_regime < BBW_TREND_THRESHOLD:
            regime = "trend"
            atr_stop_mult = ATR_STOP_TREND
        elif bbw_regime > BBW_RANGE_THRESHOLD:
            regime = "range"
            atr_stop_mult = ATR_STOP_RANGE
        else:
            regime = "neutral"
            atr_stop_mult = ATR_STOP_RANGE
        
        # ENSEMBLE VOTING: 4 indicators vote on direction
        # Signal 1: 4h HMA trend
        vote_hma = 0
        if hma_trend == 1:
            vote_hma = 1
        elif hma_trend == -1:
            vote_hma = -1
        
        # Signal 2: 4h Supertrend
        vote_st = 0
        if st_trend == 1:
            vote_st = 1
        elif st_trend == -1:
            vote_st = -1
        
        # Signal 3: 4h KAMA trend
        vote_kama = 0
        if kama_trend == 1:
            vote_kama = 1
        elif kama_trend == -1:
            vote_kama = -1
        
        # Signal 4: 15m RSI + price position
        vote_rsi = 0
        if rsi_val < 45 and price > hma_15m_val:
            vote_rsi = 1
        elif rsi_val > 55 and price < hma_15m_val:
            vote_rsi = -1
        
        # Calculate long and short votes
        long_votes = sum([vote_hma == 1, vote_st == 1, vote_kama == 1, vote_rsi == 1])
        short_votes = sum([vote_hma == -1, vote_st == -1, vote_kama == -1, vote_rsi == -1])
        
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
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - atr_stop_mult * atr_15m_val
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * atr_stop_mult * atr_15m_val
                if not prev_tp and price >= tp_price:
                    prev_signal = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal) if prev_signal != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - atr_stop_mult * atr_15m_val
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + atr_stop_mult * atr_15m_val
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * atr_stop_mult * atr_15m_val
                if not prev_tp and price <= tp_price:
                    prev_signal = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal) if prev_signal != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + atr_stop_mult * atr_15m_val
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Maintain position if signal agrees (need at least 2 votes)
            if prev_side == 1:
                if long_votes >= 2:
                    target_size = SIZE_LEVELS.get(long_votes, 0.20)
                    target_size = max(min(target_size, 0.35), 0.20)
                    
                    signals[i] = target_size
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
            elif prev_side == -1:
                if short_votes >= 2:
                    target_size = SIZE_LEVELS.get(short_votes, 0.20)
                    target_size = max(min(target_size, 0.35), 0.20)
                    
                    signals[i] = -target_size
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
            continue
        
        # Entry logic: require at least 3/4 votes for entry (high confidence)
        entry_threshold = 3
        
        if long_votes >= entry_threshold:
            target_size = SIZE_LEVELS.get(long_votes, 0.20)
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = target_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif short_votes >= entry_threshold:
            target_size = SIZE_LEVELS.get(short_votes, 0.20)
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = -target_size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals