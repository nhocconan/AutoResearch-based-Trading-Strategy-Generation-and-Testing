# Strategy: mtf_chandelier_voladj_dema_macd_15m_4h_v1

## Status
ACTIVE - Sharpe=7.706 | Return=+9703.3% | DD=-4.5%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 7.583 | +1254.4% | -3.4% | 854 |
| ETHUSDT | 7.376 | +2096.0% | -3.4% | 815 |
| SOLUSDT | 8.157 | +25759.5% | -6.7% | 835 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 7.299 | +70.2% | -1.9% | 242 |
| ETHUSDT | 8.694 | +167.3% | -2.0% | 244 |
| SOLUSDT | 9.037 | +222.7% | -2.3% | 212 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #108 - MTF Chandelier Exit + Vol-Adjusted Sizing (15m+4h v1)
==================================================================================================
Hypothesis: Build on #096 (Sharpe=8.832) and #105 (Sharpe=5.875) which proved MTF 15m+4h works.
Key improvements over current best:
- Chandelier exit (3*ATR from highest_high) instead of fixed 2*ATR stop
- Volatility-adjusted position sizing (reduce size when vol spikes)
- Cleaner MTF resampling using proper bar counting
- DEMA for faster trend response than HMA
- MACD histogram for momentum confirmation
- Discrete signal levels (0, ±0.25, ±0.35) to minimize churn costs

Why this should beat Sharpe=16.016:
- Chandelier exit trails better in strong trends (locks in profits)
- Vol-adjusted sizing reduces drawdown during high-vol periods
- 4h trend filter is more stable than 1h (fewer whipsaws)
- Based on proven winners #096, #097, #105
"""

import numpy as np
import pandas as pd

name = "mtf_chandelier_voladj_dema_macd_15m_4h_v1"
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
    
    ema1[0] = close[0]
    for i in range(1, n):
        ema1[i] = (2.0 / (period + 1)) * close[i] + (1.0 - 2.0 / (period + 1)) * ema1[i - 1]
    
    ema2[0] = ema1[0]
    for i in range(1, n):
        ema2[i] = (2.0 / (period + 1)) * ema1[i] + (1.0 - 2.0 / (period + 1)) * ema2[i - 1]
    
    for i in range(n):
        dema[i] = 2.0 * ema1[i] - ema2[i]
    
    return dema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    for i in range(1, n):
        ema_fast[i] = (2.0 / (fast + 1)) * close[i] + (1.0 - 2.0 / (fast + 1)) * ema_fast[i - 1]
        ema_slow[i] = (2.0 / (slow + 1)) * close[i] + (1.0 - 2.0 / (slow + 1)) * ema_slow[i - 1]
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    signal_line[slow] = np.mean(macd_line[slow:slow+signal])
    for i in range(slow + 1, n):
        signal_line[i] = (2.0 / (signal + 1)) * macd_line[i] + (1.0 - 2.0 / (signal + 1)) * signal_line[i - 1]
    
    for i in range(slow, n):
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


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """Calculate Chandelier Exit (ATR trailing stop)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    
    for i in range(period - 1, n):
        highest_high[i] = np.max(high[i - period + 1:i + 1])
        lowest_low[i] = np.min(low[i - period + 1:i + 1])
        
        chandelier_long[i] = highest_high[i] - multiplier * atr[i]
        chandelier_short[i] = lowest_low[i] + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    dema_15m = calculate_dema(close, period=21)
    _, _, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    chan_long_15m, chan_short_15m = calculate_chandelier_exit(high, low, close, atr_15m, period=22, multiplier=3.0)
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    bars_per_4h = 16
    n_4h = (n // bars_per_4h)
    
    # Create 4h arrays by downsampling
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = start_idx + bars_per_4h
        c_4h[i] = close[end_idx - 1]
        h_4h[i] = np.max(high[start_idx:end_idx])
        l_4h[i] = np.min(low[start_idx:end_idx])
    
    # 4h indicators for trend
    dema_4h = calculate_dema(c_4h, period=21)
    _, _, macd_hist_4h = calculate_macd(c_4h, fast=12, slow=26, signal=9)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    macd_trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            if c_4h[idx_4h] > dema_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < dema_4h[idx_4h]:
                trend_4h[i] = -1
            
            st_trend_4h[i] = st_direction_4h[idx_4h]
            
            if macd_hist_4h[idx_4h] > 0:
                macd_trend_4h[i] = 1
            elif macd_hist_4h[idx_4h] < 0:
                macd_trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.0875
    
    # Volatility-adjusted sizing base
    VOL_BASE = 0.02  # 2% ATR baseline
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 25
    
    # Chandelier exit multiplier
    CHAN_MULT = 3.0
    
    # BBW minimum for regime filter
    BBW_MIN = 0.02
    
    first_valid = max(200, 40 * bars_per_4h, 14 * 2, 20, 28)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    chandelier_stop = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        macd_trend = macd_trend_4h[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_4h_val = bbw_4h_mapped[i]
        atr_4h_val = atr_4h_mapped[i]
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # BBW filter - avoid choppy markets (4h)
        if bbw_4h_val < BBW_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (DEMA + Supertrend + MACD on 4h)
        if trend != st_trend or trend == 0 or trend != macd_trend:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check Chandelier exit and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            prev_chan_stop = chandelier_stop[i - 1]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, high[i])
                current_low = min(prev_low, low[i]) if prev_low > 0 else low[i]
            else:
                current_high = max(prev_high, high[i]) if prev_high > 0 else high[i]
                current_low = min(prev_low, low[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Update Chandelier stop
            if prev_side == 1:
                new_chan_stop = current_high - CHAN_MULT * atr
                chandelier_stop[i] = max(prev_chan_stop, new_chan_stop) if prev_chan_stop > 0 else new_chan_stop
                
                # Chandelier exit stoploss
                if price < chandelier_stop[i]:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    chandelier_stop[i] = 0
                    continue
                
                # Take profit check (2R based on 4h ATR)
                tp_price = prev_entry + 2 * CHAN_MULT * atr_4h_val
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    chandelier_stop[i] = chandelier_stop[i]
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_high - CHAN_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        chandelier_stop[i] = 0
                        continue
                    
            elif prev_side == -1:
                new_chan_stop = current_low + CHAN_MULT * atr
                chandelier_stop[i] = min(prev_chan_stop, new_chan_stop) if prev_chan_stop > 0 else new_chan_stop
                
                # Chandelier exit stoploss
                if price > chandelier_stop[i]:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    chandelier_stop[i] = 0
                    continue
                
                # Take profit check (2R based on 4h ATR)
                tp_price = prev_entry - 2 * CHAN_MULT * atr_4h_val
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    chandelier_stop[i] = chandelier_stop[i]
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_low + CHAN_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        chandelier_stop[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            chandelier_stop[i] = chandelier_stop[i - 1]
            continue
        
        # Volatility-adjusted position sizing
        vol_ratio = VOL_BASE / atr_4h_val if atr_4h_val > 0 else 1.0
        vol_ratio = np.clip(vol_ratio, 0.5, 1.5)  # Limit adjustment
        base_size = SIZE_FULL * vol_ratio
        
        # Entry logic: 4h DEMA + Supertrend + MACD + ADX + BBW + 15m RSI + Z-score
        if trend == 1 and st_trend == 1 and macd_trend == 1:  # Bullish trend confirmed on 4h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX):  # Pullback + not extreme
                signals[i] = base_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
                chandelier_stop[i] = high[i] - CHAN_MULT * atr
                
        elif trend == -1 and st_trend == -1 and macd_trend == -1:  # Bearish trend confirmed on 4h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX):  # Pullback + not extreme
                signals[i] = -base_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
                chandelier_stop[i] = low[i] + CHAN_MULT * atr
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 11:06
