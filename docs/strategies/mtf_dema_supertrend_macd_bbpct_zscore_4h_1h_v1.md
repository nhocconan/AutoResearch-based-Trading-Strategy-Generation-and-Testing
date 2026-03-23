# Strategy: mtf_dema_supertrend_macd_bbpct_zscore_4h_1h_v1

## Status
ACTIVE - Sharpe=0.213 | Return=+64.7% | DD=-24.4%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.124 | +14.5% | -16.5% | 77 |
| ETHUSDT | -0.422 | -3.3% | -22.6% | 135 |
| SOLUSDT | 1.186 | +182.9% | -33.9% | 3 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 0.064 | +6.2% | -5.1% | 81 |
| ETHUSDT | -1.288 | -9.5% | -15.9% | 57 |
| SOLUSDT | -0.378 | -0.5% | -16.5% | 31 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #005 - MTF DEMA+Supertrend+MACD+BBPct+Zscore (4h+1h v1)
==================================================================================================
Hypothesis: Current best uses 4h HMA + 1h RSI. Let's try DEMA (faster than HMA) + MACD histogram
for momentum confirmation + Bollinger %B for entry timing instead of RSI.

Key changes from #040:
- Timeframe: 1h entries + 4h trend (proven MTF combo, different from 15m+1h)
- Trend: DEMA(8/21) crossover + Supertrend(10,3) on 4h
- Entry: MACD histogram cross + BB %B (0.2-0.8 range) on 1h
- Filter: Z-score(20) < 2.0 for regime
- Position size: 0.30 (slightly conservative)
- Stoploss: 2.0*ATR trailing

Why this should work:
- DEMA responds faster than HMA to trend changes
- MACD histogram adds momentum confirmation (different from RSI)
- BB %B gives better entry timing in trending markets
- 4h trend filter reduces whipsaws vs 1h-only strategies
"""

import numpy as np
import pandas as pd

name = "mtf_dema_supertrend_macd_bbpct_zscore_4h_1h_v1"
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


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    multiplier = 2.0 / (period + 1)
    
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    ema1 = calculate_ema(close, period)
    ema2 = calculate_ema(ema1, period)
    
    dema = np.zeros(n)
    for i in range(period * 2 - 1, n):
        dema[i] = 2 * ema1[i] - ema2[i]
    
    return dema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = np.zeros(n)
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    signal_line = calculate_ema(macd_line, signal)
    
    histogram = np.zeros(n)
    for i in range(slow + signal - 1, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and %B"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbpct = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if upper[i] != lower[i]:
            bbpct[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
        else:
            bbpct[i] = 0.5
    
    return upper, middle, lower, bbpct


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


def resample_to_higher_tf(prices, tf='4h'):
    """Resample prices to higher timeframe using actual timestamps"""
    prices_indexed = prices.set_index('open_time')
    
    df_resampled = prices_indexed.resample(tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_resampled


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Resample to 4h for trend filters
    prices_df = prices.copy()
    prices_df['open_time'] = pd.to_datetime(prices_df['open_time'])
    
    try:
        df_4h = resample_to_higher_tf(prices_df, '4h')
    except Exception:
        # Fallback: simple downsampling if resample fails
        bars_per_4h = 4  # 4 x 1h = 4h
        n_4h = n // bars_per_4h
        
        c_4h = np.array([close[i * bars_per_4h + bars_per_4h - 1] for i in range(n_4h)])
        h_4h = np.array([np.max(high[i * bars_per_4h:i * bars_per_4h + bars_per_4h]) for i in range(n_4h)])
        l_4h = np.array([np.min(low[i * bars_per_4h:i * bars_per_4h + bars_per_4h]) for i in range(n_4h)])
        
        df_4h = pd.DataFrame({
            'open': c_4h,
            'high': h_4h,
            'low': l_4h,
            'close': c_4h,
            'volume': np.ones(n_4h)
        })
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    _, _, _, bbpct_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    _, _, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # 4h indicators for trend
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    n_4h = len(c_4h)
    
    dema_fast_4h = calculate_dema(c_4h, period=8)
    dema_slow_4h = calculate_dema(c_4h, period=21)
    _, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h indicators back to 1h timeframe using ffill
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    
    for i in range(n):
        # Find which 4h bar this 1h bar belongs to
        idx_4h = min(i // 4, n_4h - 1)
        if idx_4h >= 21:  # Need enough data for DEMA
            if c_4h[idx_4h] > dema_fast_4h[idx_4h] and dema_fast_4h[idx_4h] > dema_slow_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < dema_fast_4h[idx_4h] and dema_fast_4h[idx_4h] < dema_slow_4h[idx_4h]:
                trend_4h[i] = -1
            
            st_trend_4h[i] = st_direction_4h[idx_4h]
    
    # Entry thresholds
    BBPCT_LONG_MIN = 0.2
    BBPCT_LONG_MAX = 0.7
    BBPCT_SHORT_MIN = 0.3
    BBPCT_SHORT_MAX = 0.8
    ZSCORE_MAX = 2.0
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 21 * 4, 26 + 9, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(bbpct_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        zscore_val = zscore_1h[i]
        bbpct_val = bbpct_1h[i]
        macd_hist = macd_hist_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
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
                    signals[i] = SIZE_HALF
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
                    signals[i] = -SIZE_HALF
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h DEMA + Supertrend trend + 1h MACD + BB %B + Z-score
        if trend == 1 and st_trend == 1:  # Bullish trend confirmed on 4h
            if (BBPCT_LONG_MIN <= bbpct_val <= BBPCT_LONG_MAX and
                abs(zscore_val) < ZSCORE_MAX and
                macd_hist > 0):  # BB pullback + not extreme + momentum
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and st_trend == -1:  # Bearish trend confirmed on 4h
            if (BBPCT_SHORT_MIN <= bbpct_val <= BBPCT_SHORT_MAX and
                abs(zscore_val) < ZSCORE_MAX and
                macd_hist < 0):  # BB pullback + not extreme + momentum
                signals[i] = -SIZE_FULL
                position_side[i] = -1
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
2026-03-21 12:10
