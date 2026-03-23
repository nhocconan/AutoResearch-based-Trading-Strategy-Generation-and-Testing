# Strategy: clean_mtf_ensemble_regime_1h_4h_v2

## Status
ACTIVE - Sharpe=8.046 | Return=+1374978.9% | DD=-8.8%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 7.650 | +16199.1% | -6.5% | 4203 |
| ETHUSDT | 8.022 | +74529.2% | -8.5% | 4218 |
| SOLUSDT | 8.467 | +4034208.3% | -11.4% | 4236 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 8.145 | +210.6% | -3.8% | 1178 |
| ETHUSDT | 8.354 | +504.4% | -4.0% | 1245 |
| SOLUSDT | 9.765 | +1088.1% | -3.9% | 1193 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #095 - CLEAN MULTI-TF ENSEMBLE WITH REGIME FILTER (1h+4h v2)
==================================================================================================
Hypothesis: #090 achieved Sharpe=8.385 using 1h+4h regime+ensemble. Current #094 has Sharpe=0.231.
The key difference: #094 has stateful position tracking INSIDE generate_signals which is WRONG.

CRITICAL FIX: generate_signals should ONLY output position size at each bar.
- NO stoploss/TP logic here (backtest engine handles that)
- NO position state tracking (signals must be stateless)
- Signal at bar t → fill at bar t+1 open

Key innovations for #095:
1. 1h entries + 4h trend filter (proven in #090, not 15m+1h)
2. CLEAN signals - no stateful position tracking
3. 4-signal ensemble: HMA trend, Supertrend, RSI momentum, Z-score mean-reversion
4. Regime detection via BBW percentile on 4h
5. Discrete signal levels (0, ±0.20, ±0.35) to minimize churn costs
6. Signal hysteresis: require 2 consecutive bars with same vote before entering

Why this should beat #094:
- Removes buggy position tracking from signal generation
- Uses proven 1h+4h combination from #090
- Cleaner ensemble voting with proper hysteresis
- Lower signal churn = lower fee drag
"""

import numpy as np
import pandas as pd

name = "clean_mtf_ensemble_regime_1h_4h_v2"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def resample_to_higher_tf(close, high, low, tf_ratio=4):
    """Resample 1h data to 4h (4 bars per 4h candle)"""
    n = len(close)
    n_htf = n // tf_ratio
    
    c_htf = np.zeros(n_htf)
    h_htf = np.zeros(n_htf)
    l_htf = np.zeros(n_htf)
    
    for i in range(n_htf):
        start_idx = i * tf_ratio
        end_idx = start_idx + tf_ratio
        c_htf[i] = close[end_idx - 1]
        h_htf[i] = np.max(high[start_idx:end_idx])
        l_htf[i] = np.min(low[start_idx:end_idx])
    
    return c_htf, h_htf, l_htf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    hma_1h = calculate_hma(close, period=21)
    supertrend_1h, st_direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Resample to 4h for trend filter (4 x 1h = 4h)
    bars_per_4h = 4
    c_4h, h_4h, l_4h = resample_to_higher_tf(close, high, low, tf_ratio=bars_per_4h)
    
    # 4h indicators for trend
    hma_4h = calculate_hma(c_4h, period=21)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # Map 4h indicators back to 1h timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    bbw_pct_4h_mapped = np.zeros(n)
    kama_trend_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < len(c_4h) and idx_4h >= 100:
            # HMA trend
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Supertrend direction
            st_trend_4h[i] = st_direction_4h[idx_4h]
            
            # BBW percentile for regime
            bbw_pct_4h_mapped[i] = bbw_pct_4h[idx_4h]
            
            # KAMA trend
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                kama_trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                kama_trend_4h[i] = -1
    
    # Generate signals with ensemble voting
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on confidence
    SIZE_LOW = 0.20    # 2 signals agree
    SIZE_MED = 0.28    # 3 signals agree
    SIZE_HIGH = 0.35   # 4 signals agree (max)
    
    # Regime thresholds (BBW percentile on 4h)
    REGIME_LOW_VOL = 0.30   # Below this = trend following regime
    REGIME_HIGH_VOL = 0.70  # Above this = mean reversion regime
    
    # Hysteresis counter for signal stability
    prev_vote_direction = 0
    consecutive_votes = 0
    
    first_valid = max(200, 100 * bars_per_4h, 14 * 2, 20, 100)
    
    for i in range(first_valid, n):
        # Skip if any indicator is invalid
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            consecutive_votes = 0
            prev_vote_direction = 0
            continue
        
        # Get indicator values
        trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        kama_trend = kama_trend_4h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        bbw_pct = bbw_pct_4h_mapped[i]
        
        # 1h trend signals
        hma_1h_trend = 1 if close[i] > hma_1h[i] else (-1 if close[i] < hma_1h[i] else 0)
        st_1h_trend = st_direction_1h[i]
        kama_1h_trend = 1 if close[i] > kama_1h[i] else (-1 if close[i] < kama_1h[i] else 0)
        
        # Determine regime
        if bbw_pct < REGIME_LOW_VOL:
            regime = "trend"
        elif bbw_pct > REGIME_HIGH_VOL:
            regime = "mean_revert"
        else:
            regime = "neutral"
        
        # ENSEMBLE VOTING: 4 independent signals
        vote_count_long = 0
        vote_count_short = 0
        
        # Signal 1: 4h HMA trend
        if trend == 1:
            vote_count_long += 1
        elif trend == -1:
            vote_count_short += 1
        
        # Signal 2: 4h Supertrend
        if st_trend == 1:
            vote_count_long += 1
        elif st_trend == -1:
            vote_count_short += 1
        
        # Signal 3: 4h KAMA trend
        if kama_trend == 1:
            vote_count_long += 1
        elif kama_trend == -1:
            vote_count_short += 1
        
        # Signal 4: 1h RSI momentum (regime-dependent)
        if regime == "trend" or regime == "neutral":
            if rsi_val > 55:
                vote_count_long += 1
            elif rsi_val < 45:
                vote_count_short += 1
        elif regime == "mean_revert":
            if rsi_val < 35:
                vote_count_long += 1
            elif rsi_val > 65:
                vote_count_short += 1
        
        # Add 1h trend confirmation (bonus vote if aligned with 4h)
        if regime == "trend":
            if hma_1h_trend == 1 and trend == 1:
                vote_count_long += 0.5
            elif hma_1h_trend == -1 and trend == -1:
                vote_count_short += 0.5
        
        # Determine net vote
        if vote_count_long > vote_count_short:
            current_vote_direction = 1
            total_votes = vote_count_long
        elif vote_count_short > vote_count_long:
            current_vote_direction = -1
            total_votes = vote_count_short
        else:
            current_vote_direction = 0
            total_votes = 0
        
        # Hysteresis: require 2 consecutive bars with same direction
        if current_vote_direction != 0 and current_vote_direction == prev_vote_direction:
            consecutive_votes += 1
        elif current_vote_direction != 0:
            consecutive_votes = 1
            prev_vote_direction = current_vote_direction
        else:
            consecutive_votes = 0
            prev_vote_direction = 0
        
        # Generate signal based on vote count and hysteresis
        if consecutive_votes >= 2 and total_votes >= 2:
            if current_vote_direction == 1:
                if total_votes >= 4:
                    signals[i] = SIZE_HIGH
                elif total_votes >= 3:
                    signals[i] = SIZE_MED
                else:
                    signals[i] = SIZE_LOW
            else:
                if total_votes >= 4:
                    signals[i] = -SIZE_HIGH
                elif total_votes >= 3:
                    signals[i] = -SIZE_MED
                else:
                    signals[i] = -SIZE_LOW
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-21 10:45
