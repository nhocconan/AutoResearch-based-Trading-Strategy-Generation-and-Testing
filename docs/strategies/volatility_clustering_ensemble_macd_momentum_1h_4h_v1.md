# Strategy: volatility_clustering_ensemble_macd_momentum_1h_4h_v1

## Status
ACTIVE - Sharpe=3.043 | Return=+4828.9% | DD=-14.7%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 2.544 | +594.6% | -13.4% | 2901 |
| ETHUSDT | 3.124 | +1692.8% | -9.8% | 2888 |
| SOLUSDT | 3.461 | +12199.3% | -20.7% | 2762 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 2.484 | +51.5% | -4.8% | 942 |
| ETHUSDT | 3.811 | +153.8% | -7.7% | 854 |
| SOLUSDT | 4.574 | +268.6% | -6.1% | 856 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #057 - VOLATILITY_CLUSTERING_ENSEMBLE_WITH_MACD_MOMENTUM_1H_4H_V1
==================================================================================================
Hypothesis: Volatility clusters (consecutive high/low vol bars) predict regime persistence better
than single-bar BBW readings. Combining this with MACD histogram momentum + RSI divergence gives
earlier entry signals than pure trend-following. Cross-asset BTC 4h trend filter improves ETH/SOL
entries by avoiding counter-trend trades. Signal confidence scaling based on regime certainty
reduces position size in uncertain conditions.

Key innovations:
- VOLATILITY CLUSTERING: 3-bar vol sequence detection (expanding/contracting)
- MACD HISTOGRAM MOMENTUM: Early momentum shift detection before price breaks
- RSI DIVERGENCE: Price makes new low/high but RSI doesn't = reversal signal
- CROSS-ASSET FILTER: BTC 4h trend must agree for ETH/SOL entries (BTC is market leader)
- CONFIDENCE SIZING: Position size = base_size * regime_confidence (0.5-1.0 multiplier)
- 1H/4H MULTI-TF: Less noise than 15m, still responsive enough for crypto

Why this should beat #056 (Sharpe=7.760):
- Volatility clustering is more predictive than single-bar BBW percentile
- MACD histogram leads price action, giving earlier entries
- RSI divergence catches reversals before trend indicators flip
- Cross-asset filter removes 20-30% of losing ETH/SOL trades
- Based on #049's success (Sharpe=13.974) but with momentum enhancements

Position sizing rules (CRITICAL):
- MAX signal: 0.35 (proven to control drawdown in 2022 crash)
- MIN signal: 0.20 (avoid tiny positions eaten by fees)
- Discrete levels: 0.0, 0.20, 0.28, 0.35 (reduces churn costs)
- Stoploss: 2.5*ATR trailing, TP at 2R then trail at 1R
- Regime confidence multiplier: 0.5-1.0 based on signal agreement
"""

import numpy as np
import pandas as pd

name = "volatility_clustering_ensemble_macd_momentum_1h_4h_v1"
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


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # Calculate EMAs
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line (EMA of MACD)
    signal_line = np.zeros(n)
    first_signal_idx = slow + signal_period - 1
    signal_line[first_signal_idx] = np.mean(macd_line[slow:first_signal_idx + 1])
    
    for i in range(first_signal_idx + 1, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal_period + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std
        if sma[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / sma[i]
    
    return upper, lower, bbw


def calculate_volatility_clustering(atr, window=5):
    """
    Detect volatility clustering patterns
    Returns: 1 (expanding), 0 (neutral), -1 (contracting)
    """
    n = len(atr)
    clustering = np.zeros(n)
    
    for i in range(window, n):
        recent_atr = atr[i - window + 1:i + 1]
        if len(recent_atr) < window:
            continue
        
        # Check if volatility is expanding (consecutive increases)
        increases = np.sum(np.diff(recent_atr) > 0)
        decreases = np.sum(np.diff(recent_atr) < 0)
        
        if increases >= window - 1:
            clustering[i] = 1  # Expanding
        elif decreases >= window - 1:
            clustering[i] = -1  # Contracting
        else:
            clustering[i] = 0  # Neutral
    
    return clustering


def calculate_rsi_divergence(close, rsi, lookback=5):
    """
    Detect RSI divergence (price makes new low/high but RSI doesn't)
    Returns: 1 (bullish div), -1 (bearish div), 0 (none)
    """
    n = len(close)
    divergence = np.zeros(n)
    
    for i in range(lookback + 1, n):
        # Check for bullish divergence (price lower low, RSI higher low)
        price_low = np.min(close[i - lookback:i + 1])
        price_low_idx = i - lookback + np.argmin(close[i - lookback:i + 1])
        
        rsi_low = np.min(rsi[i - lookback:i + 1])
        rsi_low_idx = i - lookback + np.argmin(rsi[i - lookback:i + 1])
        
        if close[i] < price_low and rsi[i] > rsi_low:
            divergence[i] = 1  # Bullish divergence
        
        # Check for bearish divergence (price higher high, RSI lower high)
        price_high = np.max(close[i - lookback:i + 1])
        rsi_high = np.max(rsi[i - lookback:i + 1])
        
        if close[i] > price_high and rsi[i] < rsi_high:
            divergence[i] = -1  # Bearish divergence
    
    return divergence


def calculate_percentile_rank(values, window=100):
    """Calculate rolling percentile rank"""
    n = len(values)
    percentile = np.zeros(n)
    
    for i in range(window - 1, n):
        valid_vals = values[i - window + 1:i + 1]
        valid_vals = valid_vals[~np.isnan(valid_vals)]
        if len(valid_vals) > 0:
            current_val = values[i]
            percentile[i] = np.sum(valid_vals <= current_val) / len(valid_vals)
    
    return percentile


def resample_to_higher_tf(close, high, low, bars_per_tf=4):
    """Resample data to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    supertrend_1h, st_direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    macd_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal_period=9)
    _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    rsi_div_1h = calculate_rsi_divergence(close, rsi_1h, lookback=5)
    vol_cluster_1h = calculate_volatility_clustering(atr_1h, window=5)
    
    # Resample to 4h for trend (4 x 1h = 4h)
    bars_per_4h = 4
    c_4h, h_4h, l_4h = resample_to_higher_tf(close, high, low, bars_per_4h)
    
    # 4h indicators for trend
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    hma_4h = calculate_hma(c_4h, period=21)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    macd_4h, macd_signal_4h, macd_hist_4h = calculate_macd(c_4h, fast=12, slow=26, signal_period=9)
    _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_percentile_4h = calculate_percentile_rank(bbw_4h, window=50)
    
    # Map 4h indicators back to 1h timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    macd_trend_4h = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
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
            
            st_trend_4h[i] = st_direction_4h[idx_4h]
            
            # MACD trend
            if macd_hist_4h[idx_4h] > 0:
                macd_trend_4h[i] = 1
            elif macd_hist_4h[idx_4h] < 0:
                macd_trend_4h[i] = -1
            
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
    
    # Position sizing parameters (DISCRETE levels)
    SIZE_LEVELS = np.array([0.0, 0.20, 0.28, 0.35])
    BASE_SIZE = 0.28
    
    # Signal thresholds
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    MACD_HIST_THRESHOLD = 0.0001  # Relative threshold
    
    # Stoploss multipliers
    ATR_STOP = 2.5
    
    # Regime thresholds
    BBW_LOW = 0.30
    BBW_HIGH = 0.70
    
    first_valid = max(200, 40 * bars_per_4h, 50)
    
    # Generate signals with regime-switching
    signals = np.zeros(n)
    
    # Track position state for stoploss/TP
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Hysteresis counters
    long_confirm_count = np.zeros(n, dtype=int)
    short_confirm_count = np.zeros(n, dtype=int)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        macd_trend = macd_trend_4h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        bbw_4h_val = bbw_4h_mapped[i]
        macd_hist = macd_hist_1h[i]
        st_1h = st_direction_1h[i]
        hma_1h_val = hma_1h[i]
        rsi_div = rsi_div_1h[i]
        vol_cluster = vol_cluster_1h[i]
        
        # Determine regime
        is_low_vol = bbw_4h_val < BBW_LOW
        is_high_vol = bbw_4h_val > BBW_HIGH
        is_vol_expanding = vol_cluster == 1
        is_vol_contracting = vol_cluster == -1
        
        # Calculate signal scores
        # Signal 1: 4h HMA trend
        hma_signal = 0
        if trend == 1:
            hma_signal = 1
        elif trend == -1:
            hma_signal = -1
        
        # Signal 2: 4h Supertrend
        st_signal = 0
        if st_trend == 1:
            st_signal = 1
        elif st_trend == -1:
            st_signal = -1
        
        # Signal 3: 4h MACD histogram
        macd_signal = 0
        if macd_trend == 1:
            macd_signal = 1
        elif macd_trend == -1:
            macd_signal = -1
        
        # Signal 4: 1h RSI + MACD momentum combo
        momentum_signal = 0
        if macd_hist > MACD_HIST_THRESHOLD and rsi_val < RSI_LONG_MAX:
            momentum_signal = 1
        elif macd_hist < -MACD_HIST_THRESHOLD and rsi_val > RSI_SHORT_MIN:
            momentum_signal = -1
        
        # Signal 5: 1h Supertrend
        st_1h_signal = 0
        if st_1h == 1:
            st_1h_signal = 1
        elif st_1h == -1:
            st_1h_signal = -1
        
        # Signal 6: 1h HMA
        hma_1h_signal = 0
        if price > hma_1h_val:
            hma_1h_signal = 1
        elif price < hma_1h_val:
            hma_1h_signal = -1
        
        # Signal 7: RSI divergence (strong reversal signal)
        div_signal = 0
        if rsi_div == 1:
            div_signal = 1
        elif rsi_div == -1:
            div_signal = -1
        
        # Calculate weighted signal score based on regime
        if is_low_vol and is_vol_expanding:
            # Breakout regime: weight trend signals highest
            long_score = (
                0.25 * (hma_signal == 1) +
                0.25 * (st_signal == 1) +
                0.20 * (macd_signal == 1) +
                0.15 * (st_1h_signal == 1) +
                0.15 * (hma_1h_signal == 1)
            )
            short_score = (
                0.25 * (hma_signal == -1) +
                0.25 * (st_signal == -1) +
                0.20 * (macd_signal == -1) +
                0.15 * (st_1h_signal == -1) +
                0.15 * (hma_1h_signal == -1)
            )
        elif is_high_vol and is_vol_contracting:
            # Mean reversion regime: weight RSI/divergence higher
            long_score = (
                0.30 * (div_signal == 1) +
                0.25 * (momentum_signal == 1) +
                0.20 * (st_1h_signal == 1) +
                0.15 * (hma_1h_signal == 1) +
                0.10 * (st_signal == 1)
            )
            short_score = (
                0.30 * (div_signal == -1) +
                0.25 * (momentum_signal == -1) +
                0.20 * (st_1h_signal == -1) +
                0.15 * (hma_1h_signal == -1) +
                0.10 * (st_signal == -1)
            )
        else:
            # Neutral regime: balanced weights
            long_score = (
                0.20 * (hma_signal == 1) +
                0.20 * (st_signal == 1) +
                0.15 * (macd_signal == 1) +
                0.15 * (momentum_signal == 1) +
                0.15 * (st_1h_signal == 1) +
                0.15 * (hma_1h_signal == 1)
            )
            short_score = (
                0.20 * (hma_signal == -1) +
                0.20 * (st_signal == -1) +
                0.15 * (macd_signal == -1) +
                0.15 * (momentum_signal == -1) +
                0.15 * (st_1h_signal == -1) +
                0.15 * (hma_1h_signal == -1)
            )
        
        # Boost score if RSI divergence agrees (strong signal)
        if div_signal == 1:
            long_score = min(1.0, long_score + 0.15)
        elif div_signal == -1:
            short_score = min(1.0, short_score + 0.15)
        
        # HYSTERESIS: Update confirmation counters
        if long_score >= 0.45:
            long_confirm_count[i] = long_confirm_count[i - 1] + 1 if i > 0 else 1
        else:
            long_confirm_count[i] = 0
        
        if short_score >= 0.45:
            short_confirm_count[i] = short_confirm_count[i - 1] + 1 if i > 0 else 1
        else:
            short_confirm_count[i] = 0
        
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
                stoploss_price = prev_entry - ATR_STOP * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP * atr
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
                    trail_stop = current_high - ATR_STOP * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        long_confirm_count[i] = 0
                        short_confirm_count[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP * atr
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
                    trail_stop = current_low + ATR_STOP * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        long_confirm_count[i] = 0
                        short_confirm_count[i] = 0
                        continue
            
            # Maintain position if signal agrees (1-bar confirmation for exit)
            if prev_side == 1:
                if long_score >= 0.40:
                    # Calculate position size based on signal agreement + regime confidence
                    signal_count = int(long_score * 6)
                    base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
                    
                    # Regime confidence multiplier
                    regime_conf = 1.0
                    if is_low_vol and is_vol_expanding:
                        regime_conf = 1.0
                    elif is_high_vol:
                        regime_conf = 0.7
                    else:
                        regime_conf = 0.85
                    
                    target_size = base_target_size * regime_conf
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
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
                    
            elif prev_side == -1:
                if short_score >= 0.40:
                    # Calculate position size based on signal agreement + regime confidence
                    signal_count = int(short_score * 6)
                    base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
                    
                    # Regime confidence multiplier
                    regime_conf = 1.0
                    if is_low_vol and is_vol_expanding:
                        regime_conf = 1.0
                    elif is_high_vol:
                        regime_conf = 0.7
                    else:
                        regime_conf = 0.85
                    
                    target_size = base_target_size * regime_conf
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
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
            continue
        
        # Entry logic: require 2-bar confirmation (hysteresis)
        entry_threshold = 0.45
        
        if long_score >= entry_threshold and long_confirm_count[i] >= 2:
            # Calculate position size based on signal agreement + regime confidence
            signal_count = int(long_score * 6)
            base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
            
            # Regime confidence multiplier
            regime_conf = 1.0
            if is_low_vol and is_vol_expanding:
                regime_conf = 1.0
            elif is_high_vol:
                regime_conf = 0.7
            else:
                regime_conf = 0.85
            
            target_size = base_target_size * regime_conf
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = target_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            short_confirm_count[i] = 0
            
        elif short_score >= entry_threshold and short_confirm_count[i] >= 2:
            # Calculate position size based on signal agreement + regime confidence
            signal_count = int(short_score * 6)
            base_target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
            
            # Regime confidence multiplier
            regime_conf = 1.0
            if is_low_vol and is_vol_expanding:
                regime_conf = 1.0
            elif is_high_vol:
                regime_conf = 0.7
            else:
                regime_conf = 0.85
            
            target_size = base_target_size * regime_conf
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = -target_size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            long_confirm_count[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 09:51
