# Strategy: kama_super_rsi_confidence_1h_4h_v1

## Status
ACTIVE - Sharpe=0.051 | Return=+28.9% | DD=-27.7%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.495 | -11.6% | -32.7% | 1977 |
| ETHUSDT | 0.263 | +38.2% | -23.0% | 2042 |
| SOLUSDT | 0.386 | +60.0% | -27.4% | 2148 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.448 | -11.9% | -20.8% | 646 |
| ETHUSDT | -0.721 | -9.3% | -21.0% | 637 |
| SOLUSDT | -0.918 | -16.0% | -29.6% | 628 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #083 - KAMA_SUPER_RSI_CONFIDENCE_1H_4H_V1
==================================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts better to regime changes than HMA/EMA.
Combine KAMA trend + Supertrend direction + RSI timing with confidence-based position sizing.
More signals agreeing = larger position (up to 0.35 max). Fewer signals = smaller position (0.15 min).

Key innovations:
- KAMA adapts to market efficiency ratio (ER) - faster in trends, slower in chop
- Supertrend provides clear stop-loss levels for risk management
- Confidence scoring: 3 signals agree = 0.35, 2 signals = 0.25, 1 signal = 0.15
- Strong signal persistence: require 2 consecutive bars with same direction
- 4h KAMA slope filter for major trend alignment
- Avoid read-only array issues by always creating new arrays with np.where()

Why this should beat #082 (Sharpe=0.594):
- KAMA responds better to volatility changes than fixed-period HMA
- Confidence-based sizing captures more alpha in high-conviction setups
- Supertrend adds explicit risk management (stop levels)
- Cleaner signal persistence logic

Position sizing:
- MAX signal: 0.35 (3 signals agree in trend regime)
- MIN signal: 0.15 (1 signal only)
- Discrete levels: 0.0, ±0.15, ±0.25, ±0.35
- leverage=1.0 (no leverage, risk controlled by position size)
"""

import numpy as np
import pandas as pd

name = "kama_super_rsi_confidence_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_sc=2/31, slow_sc=2/31):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio (ER)
    ER = |net change| / sum of absolute changes
    High ER (trending) = faster SC, Low ER (choppy) = slower SC
    """
    n = len(close)
    if n < er_period + 1:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    net_change = np.abs(close - np.roll(close, er_period))
    net_change[:er_period] = 0
    
    sum_changes = np.zeros(n)
    for i in range(er_period, n):
        sum_changes[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1))[1:])
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = sum_changes > 0
    er[mask] = net_change[mask] / sum_changes[mask]
    er = np.clip(er, 0, 1)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator
    Returns: supertrend value, direction (1=above=bullish, -1=below=bearish)
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Calculate supertrend with direction
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    supertrend[period-1] = upper_band[period-1]
    direction[period-1] = -1
    
    for i in range(period, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction


def calculate_rsi(close, period=14):
    """Calculate RSI - vectorized with proper warmup"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.ones(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    return rsi


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_bbw_percentile(close, high, low, period=20, lookback=100):
    """Calculate Bollinger Band Width percentile for regime detection"""
    n = len(close)
    if n < lookback:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = rolling_mean + 2.0 * rolling_std
    lower = rolling_mean - 2.0 * rolling_std
    bbw = (upper - lower) / rolling_mean
    bbw = np.nan_to_num(bbw, nan=0)
    
    percentile = np.zeros(n)
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def resample_ohlcv(close, high, low, bars_per_tf):
    """Resample OHLCV to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return close[-1:], high[-1:], low[-1:]
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        c_tf[i] = close[end_idx - 1]
        h_tf[i] = np.max(high[start_idx:end_idx])
        l_tf[i] = np.min(low[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf


def map_tf_to_base(tf_array, bars_per_tf, base_length):
    """Map higher timeframe array back to base timeframe"""
    n_tf = len(tf_array)
    mapped = np.zeros(base_length)
    
    for i in range(base_length):
        tf_idx = min(i // bars_per_tf, n_tf - 1)
        mapped[i] = tf_array[tf_idx]
    
    return mapped


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Position sizing based on signal confidence
    SIZE_LOW = 0.15   # 1 signal agrees
    SIZE_MED = 0.25   # 2 signals agree
    SIZE_HIGH = 0.35  # 3 signals agree
    
    # Regime thresholds
    BBW_TREND_PERCENTILE = 0.40  # Low vol = trend regime
    BBW_MR_PERCENTILE = 0.70     # High vol = mean reversion regime
    
    # RSI thresholds by regime
    RSI_TREND_LONG = 55
    RSI_TREND_SHORT = 45
    RSI_MR_LONG = 35
    RSI_MR_SHORT = 65
    
    # Timeframe: 1h base, 4h trend filter
    bars_per_4h = 4
    
    # === BASE TIMEFRAME (1h) INDICATORS ===
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_sc=2/31, slow_sc=2/31)
    st_1h, st_dir_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    bbw_pct_1h = calculate_bbw_percentile(close, high, low, period=20, lookback=100)
    
    # === 4H TIMEFRAME INDICATORS (trend filter) ===
    c_4h, h_4h, l_4h = resample_ohlcv(close, high, low, bars_per_4h)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_sc=2/31, slow_sc=2/31)
    st_4h, st_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    bbw_pct_4h = calculate_bbw_percentile(c_4h, h_4h, l_4h, period=20, lookback=100)
    
    # Map 4h indicators to 1h
    kama_4h_mapped = map_tf_to_base(kama_4h, bars_per_4h, n)
    st_dir_4h_mapped = map_tf_to_base(st_dir_4h, bars_per_4h, n)
    bbw_pct_4h_mapped = map_tf_to_base(bbw_pct_4h, bars_per_4h, n)
    
    # === REGIME DETECTION ===
    is_trend_regime = bbw_pct_4h_mapped < BBW_TREND_PERCENTILE
    is_mr_regime = bbw_pct_4h_mapped > BBW_MR_PERCENTILE
    is_neutral = (~is_trend_regime) & (~is_mr_regime)
    
    # === SIGNAL 1: KAMA TREND ===
    # Long when price > KAMA and KAMA sloping up
    # Short when price < KAMA and KAMA sloping down
    kama_signal = np.zeros(n)
    kama_valid = kama_1h > 0
    
    kama_slope = np.zeros(n)
    kama_slope[5:] = np.sign(kama_1h[5:] - kama_1h[:-5])
    
    kama_long_mask = kama_valid & (close > kama_1h) & (kama_slope > 0)
    kama_short_mask = kama_valid & (close < kama_1h) & (kama_slope < 0)
    
    kama_signal = np.where(kama_long_mask, 1, kama_signal)
    kama_signal = np.where(kama_short_mask, -1, kama_signal)
    
    # === SIGNAL 2: SUPERTREND DIRECTION ===
    # Direct from supertrend direction indicator
    super_signal = st_dir_1h.copy()
    
    # === SIGNAL 3: RSI MOMENTUM (regime-dependent) ===
    rsi_signal = np.zeros(n)
    
    # Trend regime: RSI confirms direction
    trend_long_mask = is_trend_regime & (rsi_1h > RSI_TREND_LONG)
    trend_short_mask = is_trend_regime & (rsi_1h < RSI_TREND_SHORT)
    rsi_signal = np.where(trend_long_mask, 1, rsi_signal)
    rsi_signal = np.where(trend_short_mask, -1, rsi_signal)
    
    # Mean reversion regime: RSI extremes
    mr_long_mask = is_mr_regime & (rsi_1h < RSI_MR_LONG)
    mr_short_mask = is_mr_regime & (rsi_1h > RSI_MR_SHORT)
    rsi_signal = np.where(mr_long_mask, 1, rsi_signal)
    rsi_signal = np.where(mr_short_mask, -1, rsi_signal)
    
    # Neutral regime: standard RSI
    neutral_long_mask = is_neutral & (rsi_1h > 55)
    neutral_short_mask = is_neutral & (rsi_1h < 45)
    rsi_signal = np.where(neutral_long_mask, 1, rsi_signal)
    rsi_signal = np.where(neutral_short_mask, -1, rsi_signal)
    
    # === CONFIDENCE-BASED POSITION SIZING ===
    # Count agreeing signals for long and short
    votes_long = np.zeros(n)
    votes_short = np.zeros(n)
    
    # KAMA vote (weighted by 4h trend alignment)
    kama_4h_aligned_long = (kama_signal == 1) & (st_dir_4h_mapped >= 0)
    kama_4h_aligned_short = (kama_signal == -1) & (st_dir_4h_mapped <= 0)
    votes_long = np.where(kama_4h_aligned_long, votes_long + 1, votes_long)
    votes_long = np.where((kama_signal == 1) & (~kama_4h_aligned_long), votes_long + 0.5, votes_long)
    votes_short = np.where(kama_4h_aligned_short, votes_short + 1, votes_short)
    votes_short = np.where((kama_signal == -1) & (~kama_4h_aligned_short), votes_short + 0.5, votes_short)
    
    # Supertrend vote
    votes_long = np.where(super_signal == 1, votes_long + 1, votes_long)
    votes_short = np.where(super_signal == -1, votes_short + 1, votes_short)
    
    # RSI vote
    votes_long = np.where(rsi_signal == 1, votes_long + 1, votes_long)
    votes_short = np.where(rsi_signal == -1, votes_short + 1, votes_short)
    
    # === GENERATE FINAL SIGNALS ===
    signals = np.zeros(n)
    
    # Long signals with confidence-based sizing
    long_3sig = (votes_long >= 2.5) & (votes_long > votes_short)
    long_2sig = (votes_long >= 1.5) & (votes_long < 2.5) & (votes_long > votes_short)
    long_1sig = (votes_long >= 0.5) & (votes_long < 1.5) & (votes_long > votes_short)
    
    signals = np.where(long_3sig & is_trend_regime, SIZE_HIGH, signals)
    signals = np.where(long_3sig & (~is_trend_regime), SIZE_MED, signals)
    signals = np.where(long_2sig, SIZE_MED, signals)
    signals = np.where(long_1sig, SIZE_LOW, signals)
    
    # Short signals with confidence-based sizing
    short_3sig = (votes_short >= 2.5) & (votes_short > votes_long)
    short_2sig = (votes_short >= 1.5) & (votes_short < 2.5) & (votes_short > votes_long)
    short_1sig = (votes_short >= 0.5) & (votes_short < 1.5) & (votes_short > votes_long)
    
    signals = np.where(short_3sig & is_trend_regime, -SIZE_HIGH, signals)
    signals = np.where(short_3sig & (~is_trend_regime), -SIZE_MED, signals)
    signals = np.where(short_2sig, -SIZE_MED, signals)
    signals = np.where(short_1sig, -SIZE_LOW, signals)
    
    # === ATR VOLATILITY FILTER ===
    # Reduce position size in extremely high volatility
    atr_pct = atr_1h / close
    atr_pct = np.nan_to_num(atr_pct, nan=0)
    
    valid_atr = atr_pct > 0
    if np.sum(valid_atr) > 50:
        high_vol_threshold = np.percentile(atr_pct[valid_atr], 95)
        extreme_vol_mask = atr_pct > high_vol_threshold
        signals = np.where(extreme_vol_mask, signals * 0.5, signals)
    
    # === SIGNAL PERSISTENCE ===
    # Require 2 consecutive bars with same direction to flip
    # This reduces churn and transaction costs
    persistent_signals = np.zeros(n)
    prev_signal = 0
    
    first_valid = max(100, 48 * bars_per_4h)
    
    for i in range(n):
        if i < first_valid:
            persistent_signals[i] = 0
            prev_signal = 0
        elif i < 2:
            persistent_signals[i] = 0
            prev_signal = 0
        else:
            current = signals[i]
            
            # Same direction as previous - keep it
            if current > 0 and prev_signal > 0:
                persistent_signals[i] = current
                prev_signal = current
            elif current < 0 and prev_signal < 0:
                persistent_signals[i] = current
                prev_signal = current
            # Flipping direction - require confirmation
            elif current > 0 and prev_signal <= 0:
                if signals[i-1] > 0:
                    persistent_signals[i] = current
                    prev_signal = current
                else:
                    persistent_signals[i] = prev_signal
            elif current < 0 and prev_signal >= 0:
                if signals[i-1] < 0:
                    persistent_signals[i] = current
                    prev_signal = current
                else:
                    persistent_signals[i] = prev_signal
            # Going to flat
            elif current == 0:
                persistent_signals[i] = 0
                prev_signal = 0
            else:
                persistent_signals[i] = prev_signal
    
    signals = persistent_signals
    
    # === FINAL CLEANUP ===
    # Ensure no NaN values
    signals = np.nan_to_num(signals, nan=0.0)
    
    # Clip to max position size
    signals = np.clip(signals, -SIZE_HIGH, SIZE_HIGH)
    
    return signals
```

## Last Updated
2026-03-21 10:29
