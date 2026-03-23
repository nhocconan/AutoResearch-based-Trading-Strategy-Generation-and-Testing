# Strategy: adaptive_regime_ensemble_hma_st_rsi_zscore_15m_4h_v1

## Status
ACTIVE - Sharpe=0.087 | Return=+32.4% | DD=-27.1%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.095 | +11.8% | -24.5% | 1993 |
| ETHUSDT | -0.145 | +4.2% | -25.4% | 2025 |
| SOLUSDT | 0.502 | +81.3% | -31.5% | 2135 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 0.433 | +10.7% | -6.2% | 643 |
| ETHUSDT | -0.026 | +4.0% | -12.8% | 597 |
| SOLUSDT | 1.747 | +49.6% | -8.9% | 597 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #071 - ADAPTIVE_REGIME_ENSEMBLE_HMA_ST_RSI_ZSCORE_15M_4H_V1
==================================================================================================
Hypothesis: Combining proven indicators (HMA, Supertrend, RSI, Z-score) with adaptive regime
detection and confidence-weighted ensemble voting will beat #070's Sharpe=1.256.

Why this should work:
- HMA(16/48) provides faster trend detection than KAMA with less lag
- Supertrend(ATR=10, mult=3) gives clear trend direction with built-in stop
- RSI(14) + Z-score(20) filter for mean reversion entries in high vol regime
- 15m entries + 4h trend filter is proven (experiments #058, #060, #062)
- Confidence weighting: 2-4 signals agree = 0.20 to 0.35 position size
- BBW percentile for regime: low vol = trend follow, high vol = mean revert
- Discrete signal levels (0.0, ±0.20, ±0.28, ±0.35) reduce fee churn

Key improvements from #070:
- Cleaner variable scoping (no prev_side issues)
- Proper HMA calculation (faster than KAMA, proven in best strategies)
- Supertrend for clear trend direction + stop levels
- Better regime detection using BBW percentile + ADX
- More robust edge case handling
"""

import numpy as np
import pandas as pd

name = "adaptive_regime_ensemble_hma_st_rsi_zscore_15m_4h_v1"
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
    
    # Calculate WMA for period, period/2, and sqrt(period)
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
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_full
    
    # Apply WMA with sqrt period to the raw HMA
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
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)  # 1 = uptrend, -1 = downtrend
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
    
    # Initialize
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
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_fast_15m = calculate_hma(close, period=16)
    hma_slow_15m = calculate_hma(close, period=48)
    st_15m, st_upper_15m, st_lower_15m, st_trend_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    zscore_15m = calculate_zscore(close, period=20)
    bb_upper_15m, bb_mid_15m, bb_lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    
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
    zscore_4h = calculate_zscore(c_4h, period=20)
    _, bb_mid_4h, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Calculate BBW percentile for regime detection
    bbw_valid = bbw_4h[50:]
    if len(bbw_valid) > 0:
        bbw_sorted = np.sort(bbw_valid)
    else:
        bbw_sorted = np.array([0])
    
    # Map 4h indicators back to 15m timeframe
    hma_trend_4h = np.zeros(n)
    st_trend_4h_mapped = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    zscore_4h_mapped = np.zeros(n)
    bbw_regime = np.zeros(n)  # 0 = low vol (trend), 1 = high vol (mean revert)
    
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
            
            # ADX strength
            adx_4h_mapped[i] = adx_4h[idx_4h]
            
            # Z-score
            zscore_4h_mapped[i] = zscore_4h[idx_4h]
            
            # BBW regime (percentile-based)
            bbw_idx = np.searchsorted(bbw_sorted, bbw_4h[idx_4h]) / len(bbw_sorted)
            bbw_regime[i] = 1 if bbw_idx > 0.7 else 0  # High vol = mean revert mode
    
    # Generate signals with confidence-weighted ensemble
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on confidence
    SIZE_LOW = 0.20
    SIZE_MED = 0.28
    SIZE_HIGH = 0.35
    
    # Signal thresholds
    ADX_MIN = 20
    ZSCORE_MAX = 2.0
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    ATR_STOP_MULT = 2.5
    
    first_valid = max(300, 50 * bars_per_4h)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    entry_bar = np.zeros(n, dtype=int)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Initialize current bar state from previous
        position_side[i] = position_side[i - 1]
        entry_price[i] = entry_price[i - 1]
        entry_bar[i] = entry_bar[i - 1]
        tp_triggered[i] = tp_triggered[i - 1]
        highest_since_entry[i] = highest_since_entry[i - 1]
        lowest_since_entry[i] = lowest_since_entry[i - 1]
        
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get regime info
        hma_trend = hma_trend_4h[i]
        st_trend = st_trend_4h_mapped[i]
        adx_val = adx_4h_mapped[i]
        zscore_4h_val = zscore_4h_mapped[i]
        regime = bbw_regime[i]  # 0 = trend, 1 = mean revert
        
        # Check existing positions first (stoploss, TP, trail)
        prev_side = position_side[i - 1]
        
        if prev_side != 0:
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            atr = atr_15m[i]
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = prev_low if prev_low > 0 else price
            else:
                current_high = prev_high if prev_high > 0 else price
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
                    entry_bar[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = prev_side * SIZE_LOW
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    entry_bar[i] = entry_bar[i - 1]
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        entry_bar[i] = 0
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
                    entry_bar[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = prev_side * SIZE_LOW
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    entry_bar[i] = entry_bar[i - 1]
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        entry_bar[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            continue
        
        # No existing position - check for new entry
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        price = close[i]
        
        # Count signal agreement (confidence score)
        confidence = 0
        signal_direction = 0
        
        # Signal 1: HMA trend direction (4h)
        if hma_trend != 0:
            confidence += 1
            signal_direction += hma_trend
        
        # Signal 2: Supertrend direction (4h)
        if st_trend != 0:
            confidence += 1
            signal_direction += st_trend
        
        # Signal 3: ADX trend strength
        if adx_val >= ADX_MIN:
            confidence += 1
            if hma_trend != 0:
                signal_direction += hma_trend
        
        # Signal 4: Z-score filter (not extreme)
        if abs(zscore_4h_val) < ZSCORE_MAX:
            confidence += 1
            if hma_trend != 0:
                signal_direction += hma_trend
        
        # Determine position size based on confidence
        if confidence >= 4:
            position_size = SIZE_HIGH
        elif confidence >= 3:
            position_size = SIZE_MED
        elif confidence >= 2:
            position_size = SIZE_LOW
        else:
            signals[i] = 0.0
            continue
        
        # Determine entry direction based on regime
        if regime == 0:  # Low vol - trend following
            if signal_direction >= 2 and hma_trend == 1 and st_trend == 1:
                # LONG entry
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                entry_bar[i] = i
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
            elif signal_direction <= -2 and hma_trend == -1 and st_trend == -1:
                # SHORT entry
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                entry_bar[i] = i
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                
        else:  # High vol - mean reversion
            if zscore_val < -1.5 and rsi_val < 40:
                # LONG mean reversion
                signals[i] = position_size * 0.7
                position_side[i] = 1
                entry_price[i] = price
                entry_bar[i] = i
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
            elif zscore_val > 1.5 and rsi_val > 60:
                # SHORT mean reversion
                signals[i] = -position_size * 0.7
                position_side[i] = -1
                entry_price[i] = price
                entry_bar[i] = i
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-21 10:13
