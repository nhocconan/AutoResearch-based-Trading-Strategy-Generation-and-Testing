# Strategy: simplified_mtf_hma_st_kama_rsi_1h_4h_v1

## Status
ACTIVE - Sharpe=0.589 | Return=+68.8% | DD=-13.5%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 0.474 | +44.7% | -10.2% | 277 |
| ETHUSDT | 0.412 | +45.2% | -10.7% | 299 |
| SOLUSDT | 0.880 | +116.6% | -19.6% | 317 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 1.314 | +19.1% | -3.6% | 106 |
| ETHUSDT | 0.959 | +21.7% | -8.0% | 89 |
| SOLUSDT | 1.572 | +35.3% | -7.3% | 94 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #072 - SIMPLIFIED_MTF_HMA_ST_KAMA_RSI_1H_4H_V1
==================================================================================================
Hypothesis: After #071's poor Sharpe (0.087), simplify the ensemble. Use proven 1h entries + 4h 
trend filter with fewer, higher-quality signals. Focus on HMA + Supertrend + KAMA for trend, 
RSI for pullback entries. Discrete sizing (0.20/0.28/0.35) reduces fee churn.

Why this should work:
- 1h timeframe has less noise than 15m (fewer false signals, better Sharpe)
- HMA(16/48) proven in best strategies (#060, #062)
- KAMA adds adaptive trend filtering (responds to volatility)
- Supertrend gives clear direction + stop levels
- RSI(14) for pullback entries in established trends
- 4h trend filter prevents counter-trend trades
- Simpler logic = fewer bugs (learned from #069 crash)
- Position sizing max 0.35 controls drawdown

Key changes from #071:
- 1h instead of 15m (cleaner signals)
- Fewer regime conditions (ADX + BBW only, not Z-score regime)
- Cleaner position state tracking (no prev_side scoping issues)
- Entry only on RSI pullback in trend direction (higher quality)
"""

import numpy as np
import pandas as pd

name = "simplified_mtf_hma_st_kama_rsi_1h_4h_v1"
timeframe = "1h"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        # Efficiency Ratio
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        # Smoothing constants
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_fast_1h = calculate_hma(close, period=16)
    hma_slow_1h = calculate_hma(close, period=48)
    kama_1h = calculate_kama(close, period=10)
    st_1h, st_upper_1h, st_lower_1h, st_trend_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    bb_upper_1h, bb_mid_1h, bb_lower_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # Resample to 4h for trend regime (4 x 1h = 4h)
    bars_per_4h = 4
    c_4h, h_4h, l_4h, o_4h = resample_to_timeframe(close, high, low, open_price, bars_per_4h)
    n_4h = len(c_4h)
    
    if n_4h < 50:
        return np.zeros(n)
    
    # 4h indicators for trend regime
    hma_fast_4h = calculate_hma(c_4h, period=16)
    hma_slow_4h = calculate_hma(c_4h, period=48)
    kama_4h = calculate_kama(c_4h, period=10)
    st_4h, st_upper_4h, st_lower_4h, st_trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    _, bb_mid_4h, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Calculate BBW percentile for regime detection
    bbw_valid = bbw_4h[50:]
    if len(bbw_valid) > 0:
        bbw_sorted = np.sort(bbw_valid)
    else:
        bbw_sorted = np.array([0])
    
    # Map 4h indicators back to 1h timeframe
    hma_trend_4h = np.zeros(n)
    st_trend_4h_mapped = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_regime = np.zeros(n)
    kama_trend_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 50:
            # HMA trend
            if c_4h[idx_4h] > hma_slow_4h[idx_4h] and hma_fast_4h[idx_4h] > hma_slow_4h[idx_4h]:
                hma_trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_slow_4h[idx_4h] and hma_fast_4h[idx_4h] < hma_slow_4h[idx_4h]:
                hma_trend_4h[i] = -1
            
            # KAMA trend
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                kama_trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                kama_trend_4h[i] = -1
            
            # Supertrend
            st_trend_4h_mapped[i] = st_trend_4h[idx_4h]
            
            # ADX strength
            adx_4h_mapped[i] = adx_4h[idx_4h]
            
            # BBW regime (percentile-based)
            bbw_idx = np.searchsorted(bbw_sorted, bbw_4h[idx_4h]) / len(bbw_sorted)
            bbw_regime[i] = 1 if bbw_idx > 0.7 else 0
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_LOW = 0.20
    SIZE_MED = 0.28
    SIZE_HIGH = 0.35
    
    # Thresholds
    ADX_MIN = 20
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
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
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            pos_side = 0
            pos_entry = 0.0
            pos_tp_triggered = False
            pos_highest = 0.0
            pos_lowest = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi = rsi_1h[i]
        
        # Get regime info
        hma_trend = hma_trend_4h[i]
        kama_trend = kama_trend_4h[i]
        st_trend = st_trend_4h_mapped[i]
        adx_val = adx_4h_mapped[i]
        regime = bbw_regime[i]
        
        # Manage existing position
        if pos_side != 0:
            # Update highest/lowest since entry
            if pos_side == 1:
                pos_highest = max(pos_highest, price) if pos_highest > 0 else price
                pos_lowest = pos_lowest if pos_lowest > 0 else price
            else:
                pos_highest = pos_highest if pos_highest > 0 else price
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
                    signals[i] = SIZE_LOW
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
                    signals[i] = -SIZE_LOW
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
        # Count signal agreement
        confidence = 0
        signal_direction = 0
        
        # Signal 1: HMA trend (4h)
        if hma_trend != 0:
            confidence += 1
            signal_direction += hma_trend
        
        # Signal 2: KAMA trend (4h)
        if kama_trend != 0:
            confidence += 1
            signal_direction += kama_trend
        
        # Signal 3: Supertrend (4h)
        if st_trend != 0:
            confidence += 1
            signal_direction += st_trend
        
        # Signal 4: ADX strength
        if adx_val >= ADX_MIN:
            confidence += 1
        
        # Determine position size
        if confidence >= 4:
            position_size = SIZE_HIGH
        elif confidence >= 3:
            position_size = SIZE_MED
        elif confidence >= 2:
            position_size = SIZE_LOW
        else:
            signals[i] = 0.0
            continue
        
        # Entry logic based on regime
        if regime == 0:  # Low vol - trend following
            # LONG: uptrend + RSI pullback
            if signal_direction >= 2 and hma_trend == 1 and st_trend == 1:
                if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
                    signals[i] = position_size
                    pos_side = 1
                    pos_entry = price
                    pos_entry_bar = i
                    pos_tp_triggered = False
                    pos_highest = price
                    pos_lowest = price
                else:
                    signals[i] = 0.0
            
            # SHORT: downtrend + RSI pullback
            elif signal_direction <= -2 and hma_trend == -1 and st_trend == -1:
                if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
                    signals[i] = -position_size
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
                
        else:  # High vol - be more conservative
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-21 10:14
