#!/usr/bin/env python3
"""
EXPERIMENT #074 - ENSEMBLE_REGIME_CONFIDENCE_BBW_15M_4H_V1
==================================================================================================
Hypothesis: After #073's crash (variable scope issue), return to a cleaner ensemble approach
with proper regime detection using Bollinger Band Width percentile. Use 3-signal voting system
(HMA trend, Supertrend, RSI momentum) with confidence-weighted position sizing. Regime detection
determines whether to trend-follow (low BBW) or mean-revert (high BBW).

Why this should work:
- 15m timeframe proven successful (#062 Sharpe=11.924)
- BBW regime detection is more stable than ATR percentile
- 3-signal voting reduces false entries vs single-indicator strategies
- Confidence-weighted sizing: more agreement = larger position (up to 0.35)
- 4h trend filter prevents counter-trend trades
- Discrete signal levels (0, ±0.20, ±0.28, ±0.35) minimize fee churn
- Clean variable scoping (learned from #073 crash)

Key improvements from #073:
- Fixed variable scope issues (all functions receive required parameters)
- BBW regime instead of ATR percentile (more stable)
- Simpler 3-signal voting system (HMA, Supertrend, RSI)
- Regime-aware entry logic (trend vs mean reversion)
- Cleaner position state management
"""

import numpy as np
import pandas as pd

name = "ensemble_regime_confidence_bbw_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, wma_period):
        n_wma = len(data)
        result = np.zeros(n_wma)
        weights = np.arange(1, wma_period + 1)
        weight_sum = np.sum(weights)
        
        for i in range(wma_period - 1, n_wma):
            window = data[i - wma_period + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        
        return result
    
    wma_full = wma(close, period)
    wma_half = wma(close, half_period)
    
    hma_raw = 2 * wma_half - wma_full
    
    hma = np.zeros(n)
    weights = np.arange(1, sqrt_period + 1)
    weight_sum = np.sum(weights)
    
    for i in range(sqrt_period - 1, n):
        if i >= len(hma_raw):
            break
        start_idx = max(0, i - sqrt_period + 1)
        window = hma_raw[start_idx:i + 1]
        if len(window) == sqrt_period:
            hma[i] = np.sum(window * weights) / weight_sum
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
    
    supertrend[period] = upper_band[period]
    trend[period] = -1 if close[period] < supertrend[period] else 1
    
    for i in range(period + 1, n):
        if trend[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, upper_band, lower_band, trend


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
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = np.zeros(n)
    std = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        sma[i] = np.mean(window)
        std[i] = np.std(window)
        upper[i] = sma[i] + std_mult * std[i]
        lower[i] = sma[i] - std_mult * std[i]
        if sma[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / sma[i] * 100
    
    return upper, lower, sma, bbw


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
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def resample_to_timeframe(close, high, low, open_price, bars_per_tf):
    """Resample data to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    o_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        if end_idx <= n:
            c_tf[i] = close[end_idx - 1]
            h_tf[i] = np.max(high[start_idx:end_idx])
            l_tf[i] = np.min(low[start_idx:end_idx])
            o_tf[i] = open_price[start_idx]
    
    return c_tf, h_tf, l_tf, o_tf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_price = prices["open"].values
    n = len(close)
    
    if n < 500:
        return np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_fast_15m = calculate_hma(close, period=16)
    hma_slow_15m = calculate_hma(close, period=48)
    st_15m, st_upper_15m, st_lower_15m, st_trend_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    upper_15m, lower_15m, sma_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Resample to 4h for trend regime (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h, o_4h = resample_to_timeframe(close, high, low, open_price, bars_per_4h)
    n_4h = len(c_4h)
    
    if n_4h < 50:
        return np.zeros(n)
    
    # 4h indicators for trend regime
    hma_fast_4h = calculate_hma(c_4h, period=16)
    hma_slow_4h = calculate_hma(c_4h, period=48)
    st_4h, st_upper_4h, st_lower_4h, st_trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    upper_4h, lower_4h, sma_4h, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    
    # Calculate BBW percentile for regime detection (using 4h data)
    bbw_valid = bbw_4h[50:]
    if len(bbw_valid) > 0:
        bbw_sorted = np.sort(bbw_valid[bbw_valid > 0])
    else:
        bbw_sorted = np.array([1.0])
    
    # Map 4h indicators back to 15m timeframe
    hma_trend_4h = np.zeros(n)
    st_trend_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    bbw_percentile = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    regime = np.zeros(n)  # 0 = low vol (trend), 1 = high vol (mean revert)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 50:
            # HMA trend
            if c_4h[idx_4h] > hma_slow_4h[idx_4h] and hma_fast_4h[idx_4h] > hma_slow_4h[idx_4h]:
                hma_trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_slow_4h[idx_4h] and hma_fast_4h[idx_4h] < hma_slow_4h[idx_4h]:
                hma_trend_4h[i] = -1
            
            # Supertrend
            st_trend_4h_mapped[i] = st_trend_4h[idx_4h]
            
            # BBW value and percentile
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            if len(bbw_sorted) > 0 and bbw_4h[idx_4h] > 0:
                bbw_pct = np.searchsorted(bbw_sorted, bbw_4h[idx_4h]) / len(bbw_sorted)
                bbw_percentile[i] = bbw_pct
                regime[i] = 1 if bbw_pct > 0.7 else 0
            else:
                bbw_percentile[i] = 0.5
                regime[i] = 0
            
            # ATR value
            atr_4h_mapped[i] = atr_4h[idx_4h]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_LOW = 0.20
    SIZE_MED = 0.28
    SIZE_HIGH = 0.35
    
    # Thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    RSI_MR_LONG = 30
    RSI_MR_SHORT = 70
    ATR_STOP_MULT = 2.5
    
    first_valid = max(300, 50 * bars_per_4h)
    
    # Position state tracking
    pos_side = 0
    pos_entry = 0.0
    pos_entry_bar = 0
    pos_tp_triggered = False
    pos_highest = 0.0
    pos_lowest = 0.0
    
    for i in range(first_valid, n):
        # Check for invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            if pos_side != 0:
                pos_side = 0
                pos_entry = 0.0
                pos_tp_triggered = False
                pos_highest = 0.0
                pos_lowest = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi = rsi_15m[i]
        
        # Get regime info
        hma_trend = hma_trend_4h[i]
        st_trend = st_trend_4h_mapped[i]
        bbw_pct = bbw_percentile[i]
        regime_val = regime[i]
        atr_4h_val = atr_4h_mapped[i]
        
        # Manage existing position
        if pos_side != 0:
            # Update highest/lowest since entry
            if pos_side == 1:
                pos_highest = max(pos_highest, price) if pos_highest > 0 else price
            else:
                pos_lowest = min(pos_lowest, price) if pos_lowest > 0 else price
            
            # Stoploss check
            if pos_side == 1:
                stoploss_price = pos_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    pos_side = 0
                    pos_entry = 0.0
                    pos_tp_triggered = False
                    pos_highest = 0.0
                    pos_lowest = 0.0
                    continue
                
                # Take profit (2R) - reduce to half
                tp_price = pos_entry + 2 * ATR_STOP_MULT * atr
                if not pos_tp_triggered and price >= tp_price:
                    signals[i] = SIZE_LOW * 0.5
                    pos_tp_triggered = True
                    continue
                
                # Trail stop at 1R
                if pos_tp_triggered:
                    trail_stop = pos_highest - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        pos_side = 0
                        pos_entry = 0.0
                        pos_tp_triggered = False
                        pos_highest = 0.0
                        pos_lowest = 0.0
                        continue
            
            elif pos_side == -1:
                stoploss_price = pos_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    pos_side = 0
                    pos_entry = 0.0
                    pos_tp_triggered = False
                    pos_highest = 0.0
                    pos_lowest = 0.0
                    continue
                
                # Take profit (2R) - reduce to half
                tp_price = pos_entry - 2 * ATR_STOP_MULT * atr
                if not pos_tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_LOW * 0.5
                    pos_tp_triggered = True
                    continue
                
                # Trail stop at 1R
                if pos_tp_triggered:
                    trail_stop = pos_lowest + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        pos_side = 0
                        pos_entry = 0.0
                        pos_tp_triggered = False
                        pos_highest = 0.0
                        pos_lowest = 0.0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1] if i > 0 else 0.0
            continue
        
        # No position - check for new entry
        # Count signal agreement (3-signal voting)
        confidence = 0
        signal_direction = 0
        
        # Signal 1: HMA trend (4h)
        if hma_trend != 0:
            confidence += 1
            signal_direction += hma_trend
        
        # Signal 2: Supertrend (4h)
        if st_trend != 0:
            confidence += 1
            signal_direction += int(st_trend)
        
        # Signal 3: 15m RSI momentum
        if rsi > 50:
            confidence += 1
            signal_direction += 1
        elif rsi < 50:
            confidence += 1
            signal_direction -= 1
        
        # Determine position size based on confidence
        if confidence >= 3:
            base_size = SIZE_HIGH
        elif confidence >= 2:
            base_size = SIZE_MED
        else:
            signals[i] = 0.0
            continue
        
        # Entry logic based on regime
        if regime_val == 0:
            # LOW VOLATILITY (BBW < 70th percentile) - TREND FOLLOWING
            # LONG: uptrend + RSI pullback
            if signal_direction >= 2 and hma_trend == 1 and st_trend == 1:
                if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
                    # Additional filter: price above KAMA
                    if price > kama_15m[i] and kama_15m[i] > 0:
                        signals[i] = base_size
                        pos_side = 1
                        pos_entry = price
                        pos_entry_bar = i
                        pos_tp_triggered = False
                        pos_highest = price
                        pos_lowest = price
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            
            # SHORT: downtrend + RSI pullback
            elif signal_direction <= -2 and hma_trend == -1 and st_trend == -1:
                if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
                    # Additional filter: price below KAMA
                    if price < kama_15m[i] and kama_15m[i] > 0:
                        signals[i] = -base_size
                        pos_side = -1
                        pos_entry = price
                        pos_entry_bar = i
                        pos_tp_triggered = False
                        pos_highest = price
                        pos_lowest = price
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        else:
            # HIGH VOLATILITY (BBW > 70th percentile) - MEAN REVERSION
            # LONG: RSI oversold in uptrend
            if hma_trend == 1 and rsi <= RSI_MR_LONG:
                # Additional filter: price below lower BB
                if price < lower_15m[i] and lower_15m[i] > 0:
                    signals[i] = SIZE_MED
                    pos_side = 1
                    pos_entry = price
                    pos_entry_bar = i
                    pos_tp_triggered = False
                    pos_highest = price
                    pos_lowest = price
                else:
                    signals[i] = 0.0
            # SHORT: RSI overbought in downtrend
            elif hma_trend == -1 and rsi >= RSI_MR_SHORT:
                # Additional filter: price above upper BB
                if price > upper_15m[i] and upper_15m[i] > 0:
                    signals[i] = -SIZE_MED
                    pos_side = -1
                    pos_entry = price
                    pos_entry_bar = i
                    pos_tp_triggered = False
                    pos_highest = price
                    pos_lowest = price
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals