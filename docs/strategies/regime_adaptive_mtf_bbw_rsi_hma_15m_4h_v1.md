# Strategy: regime_adaptive_mtf_bbw_rsi_hma_15m_4h_v1

## Status
ACTIVE - Sharpe=0.164 | Return=+103.9% | DD=-36.9%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.140 | -34.2% | -40.7% | 164 |
| ETHUSDT | 0.373 | +48.0% | -26.4% | 1 |
| SOLUSDT | 1.260 | +297.9% | -43.4% | 8 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.803 | -2.7% | -15.0% | 124 |
| ETHUSDT | -0.055 | +3.8% | -19.5% | 60 |
| SOLUSDT | -0.102 | +2.1% | -16.1% | 43 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #055 - Regime-Adaptive MTF with Adaptive Sizing (15m + 4h)
==================================================================================================
Hypothesis: Failed ensemble strategies (#051-#054) had too many signal changes = excessive fees.
This strategy uses regime detection (BBW percentile) to switch between mean-reversion and trend-following,
with adaptive position sizing based on signal agreement (more indicators agree = larger position).

Key improvements over #040:
- Use mtf_data helper for PROPER 4h alignment (critical for SOL data gaps)
- Regime detection: BBW percentile → mean-revert in low vol, trend-follow in high vol
- Adaptive sizing: 0.20 (1 signal), 0.275 (2 signals), 0.35 (3+ signals agree)
- Simpler entry logic to reduce churn and fees
- 4h trend filter + 15m entries (proven combination from #031, #034, #035)
- Max signal magnitude: 0.35 (critical for drawdown control)

Why this should beat #040:
- Proper MTF alignment using mtf_data helper (46 strategies failed without this)
- Regime adaptation reduces losses in wrong market conditions
- Adaptive sizing maximizes returns when confidence is high
- Fewer signal changes = lower fees
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_mtf_bbw_rsi_hma_15m_4h_v1"
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
    
    wma1 = pd.Series(close).rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    raw_hma = 2 * wma1 - wma2
    
    hma = pd.Series(raw_hma).rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
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
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
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
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    middle = rolling_mean
    upper = middle + std_mult * rolling_std
    lower = middle - std_mult * rolling_std
    
    bbw = np.zeros(n)
    for i in range(period - 1, n):
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
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend
        hma_4h = calculate_hma(close_4h, period=21)
        _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
        
        # Align 4h indicators to 15m timeframe
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = hma_15m
        st_4h_aligned = st_direction_15m
        bbw_pct_4h_aligned = bbw_pct_15m
    
    # Generate signals with regime-adaptive logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on signal agreement
    SIZE_LOW = 0.20    # 1 signal agrees
    SIZE_MED = 0.275   # 2 signals agree
    SIZE_HIGH = 0.35   # 3+ signals agree (MAX - critical for drawdown control)
    
    # RSI thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score threshold
    ZSCORE_MAX = 2.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # BBW percentile thresholds for regime
    BBW_LOW_REGIME = 0.30   # Below 30th percentile = low vol (mean revert)
    BBW_HIGH_REGIME = 0.70  # Above 70th percentile = high vol (trend follow)
    
    first_valid = max(200, 100, 14 * 2, 20, 28)
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(st_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        if close[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif close[i] < hma_4h_aligned[i]:
            trend_4h = -1
        else:
            trend_4h = 0
        
        st_4h = st_4h_aligned[i]
        bbw_regime = bbw_pct_4h_aligned[i]
        
        # Determine regime
        if bbw_regime < BBW_LOW_REGIME:
            regime = 'mean_revert'  # Low volatility
        elif bbw_regime > BBW_HIGH_REGIME:
            regime = 'trend_follow'  # High volatility
        else:
            regime = 'neutral'  # Skip trades in neutral regime
        
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        price = close[i]
        atr = atr_15m[i]
        
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
            
            # Stoploss check (2.5*ATR)
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
        
        # Skip neutral regime
        if regime == 'neutral':
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Count signal agreements for adaptive sizing
        signal_count = 0
        signal_direction = 0
        
        if regime == 'mean_revert':
            # Mean reversion: fade extremes against 4h trend
            if trend_4h == 1:  # 4h bullish, look for long pullbacks
                if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                    zscore_val < -0.5 and zscore_val > -ZSCORE_MAX):
                    signal_count += 1
                    signal_direction = 1
                if st_direction_15m[i] == 1:
                    signal_count += 1
                    signal_direction = 1
                    
            elif trend_4h == -1:  # 4h bearish, look for short bounces
                if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                    zscore_val > 0.5 and zscore_val < ZSCORE_MAX):
                    signal_count += 1
                    signal_direction = -1
                if st_direction_15m[i] == -1:
                    signal_count += 1
                    signal_direction = -1
                    
        elif regime == 'trend_follow':
            # Trend following: go with 4h trend
            if trend_4h == 1 and st_4h == 1:  # 4h bullish confirmed
                if rsi_val > 50 and rsi_val < 70:
                    signal_count += 1
                    signal_direction = 1
                if st_direction_15m[i] == 1:
                    signal_count += 1
                    signal_direction = 1
                if zscore_val > 0 and zscore_val < ZSCORE_MAX:
                    signal_count += 1
                    signal_direction = 1
                    
            elif trend_4h == -1 and st_4h == -1:  # 4h bearish confirmed
                if rsi_val < 50 and rsi_val > 30:
                    signal_count += 1
                    signal_direction = -1
                if st_direction_15m[i] == -1:
                    signal_count += 1
                    signal_direction = -1
                if zscore_val < 0 and zscore_val > -ZSCORE_MAX:
                    signal_count += 1
                    signal_direction = -1
        
        # Adaptive position sizing based on signal agreement
        if signal_count >= 3:
            position_size = SIZE_HIGH
        elif signal_count == 2:
            position_size = SIZE_MED
        elif signal_count == 1:
            position_size = SIZE_LOW
        else:
            position_size = 0
        
        # Enter position if signals agree
        if signal_count >= 1 and signal_direction != 0:
            signals[i] = signal_direction * position_size
            position_side[i] = signal_direction
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 14:12
