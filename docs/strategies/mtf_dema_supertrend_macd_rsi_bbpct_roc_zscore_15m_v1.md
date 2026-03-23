# Strategy: mtf_dema_supertrend_macd_rsi_bbpct_roc_zscore_15m_v1

## Status
ACTIVE - Sharpe=3.578 | Return=+203.7% | DD=-2.1%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 2.423 | +76.1% | -2.2% | 205 |
| ETHUSDT | 3.180 | +104.5% | -2.2% | 251 |
| SOLUSDT | 5.130 | +430.4% | -2.0% | 383 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 1.331 | +8.9% | -0.4% | 47 |
| ETHUSDT | 3.648 | +20.9% | -0.7% | 67 |
| SOLUSDT | 3.971 | +27.0% | -1.4% | 96 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #019 - MTF DEMA+Supertrend+MACD+RSI+BBPct+Z-score (15m+1h v1)
==================================================================================================
Hypothesis: Current best #012 uses 1h_4h with Sharpe=0.478. Experiments #031,#034,#035 showed 
15m+1h MTF can achieve Sharpe > 7.5 with proper filtering.

Key changes from #040:
- Replace HMA with DEMA(8/21) for faster trend response (less lag than HMA)
- Add MACD histogram for momentum confirmation (not just RSI pullback)
- Add Bollinger %B (position within bands) instead of just BBW
- Add ROC(10) for rate-of-change momentum filter
- Position size: 0.30 (slightly more conservative)
- Stoploss: 2.5*ATR (wider for 15m noise reduction)
- ADX threshold: 20 (lower than #040's 25 for more trades)

Why this should beat #012 (Sharpe=0.478):
- 15m has 4x more data points than 1h for better signal resolution
- DEMA responds faster to trend changes than HMA
- MACD histogram adds momentum confirmation layer
- Bollinger %B gives precise entry timing within volatility bands
- ROC filter ensures we only trade when momentum supports direction
- Based on proven 15m+1h MTF structure from winning experiments
"""

import numpy as np
import pandas as pd

name = "mtf_dema_supertrend_macd_rsi_bbpct_roc_zscore_15m_v1"
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


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    dema = np.zeros(n)
    
    multiplier = 2.0 / (period + 1)
    
    ema1[0] = close[0]
    for i in range(1, n):
        ema1[i] = multiplier * close[i] + (1 - multiplier) * ema1[i - 1]
    
    ema2[0] = ema1[0]
    for i in range(1, n):
        ema2[i] = multiplier * ema1[i] + (1 - multiplier) * ema2[i - 1]
    
    for i in range(n):
        dema[i] = 2 * ema1[i] - ema2[i]
    
    return dema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    fast_multiplier = 2.0 / (fast + 1)
    slow_multiplier = 2.0 / (slow + 1)
    signal_multiplier = 2.0 / (signal + 1)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    for i in range(1, n):
        ema_fast[i] = fast_multiplier * close[i] + (1 - fast_multiplier) * ema_fast[i - 1]
        ema_slow[i] = slow_multiplier * close[i] + (1 - slow_multiplier) * ema_slow[i - 1]
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    signal_line[slow] = macd_line[slow]
    for i in range(slow + 1, n):
        signal_line[i] = signal_multiplier * macd_line[i] + (1 - signal_multiplier) * signal_line[i - 1]
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


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


def calculate_roc(close, period=10):
    """Calculate Rate of Change"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    roc = np.zeros(n)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
        else:
            roc[i] = 0
    
    return roc


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    dema_fast_15m = calculate_dema(close, period=8)
    dema_slow_15m = calculate_dema(close, period=21)
    macd_line_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    _, _, _, bbpct_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    roc_15m = calculate_roc(close, period=10)
    
    # Resample to 1h for trend filters (4 x 15m = 1h)
    bars_per_1h = 4
    n_1h = (n // bars_per_1h)
    
    if n_1h < 50:
        return np.zeros(n)
    
    # Create 1h arrays by downsampling
    c_1h = np.zeros(n_1h)
    h_1h = np.zeros(n_1h)
    l_1h = np.zeros(n_1h)
    
    for i in range(n_1h):
        start_idx = i * bars_per_1h
        end_idx = start_idx + bars_per_1h
        c_1h[i] = close[end_idx - 1]
        h_1h[i] = np.max(high[start_idx:end_idx])
        l_1h[i] = np.min(low[start_idx:end_idx])
    
    # 1h indicators for trend
    dema_fast_1h = calculate_dema(c_1h, period=8)
    dema_slow_1h = calculate_dema(c_1h, period=21)
    supertrend_1h, st_direction_1h = calculate_supertrend(h_1h, l_1h, c_1h, period=10, multiplier=3.0)
    adx_1h = calculate_adx(h_1h, l_1h, c_1h, period=14)
    
    # Map 1h indicators back to 15m timeframe
    trend_1h = np.zeros(n)
    st_trend_1h = np.zeros(n)
    adx_1h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_1h = i // bars_per_1h
        if idx_1h < n_1h and idx_1h >= 30:
            if c_1h[idx_1h] > dema_fast_1h[idx_1h] and dema_fast_1h[idx_1h] > dema_slow_1h[idx_1h]:
                trend_1h[i] = 1
            elif c_1h[idx_1h] < dema_fast_1h[idx_1h] and dema_fast_1h[idx_1h] < dema_slow_1h[idx_1h]:
                trend_1h[i] = -1
            
            st_trend_1h[i] = st_direction_1h[idx_1h]
            adx_1h_mapped[i] = adx_1h[idx_1h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ADX threshold for trend strength (1h)
    ADX_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Bollinger %B thresholds
    BBPCT_LONG_MAX = 0.70
    BBPCT_SHORT_MIN = 0.30
    
    # ROC threshold for momentum confirmation
    ROC_MIN = 0.5
    
    first_valid = max(200, 40 * bars_per_1h, 35, 26, 28)
    
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
        
        trend = trend_1h[i]
        st_trend = st_trend_1h[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_1h_val = adx_1h_mapped[i]
        bbpct_val = bbpct_15m[i]
        macd_hist_val = macd_hist_15m[i]
        roc_val = roc_15m[i]
        
        # ADX filter (1h) - only trade when trend is strong enough
        if adx_1h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (DEMA crossover + Supertrend on 1h)
        if trend != st_trend or trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
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
        
        # Entry logic: 1h DEMA + Supertrend + ADX + 15m RSI + MACD + BBPct + Z-score + ROC
        if trend == 1 and st_trend == 1:  # Bullish trend confirmed on 1h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX and
                macd_hist_val > 0 and
                bbpct_val < BBPCT_LONG_MAX and
                roc_val > ROC_MIN):  # Pullback + momentum + not extreme + room in bands
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and st_trend == -1:  # Bearish trend confirmed on 1h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX and
                macd_hist_val < 0 and
                bbpct_val > BBPCT_SHORT_MIN and
                roc_val < -ROC_MIN):  # Pullback + momentum + not extreme + room in bands
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
2026-03-21 12:25
