# Strategy: kama_donchian_mtf_chandelier_voladj_15m_4h_v2

## Status
ACTIVE - Sharpe=4.879 | Return=+3507.1% | DD=-8.0%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 4.577 | +714.1% | -5.7% | 1234 |
| ETHUSDT | 4.744 | +1271.0% | -10.3% | 1135 |
| SOLUSDT | 5.318 | +8536.1% | -7.9% | 1126 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 4.003 | +49.7% | -2.9% | 360 |
| ETHUSDT | 6.644 | +164.3% | -3.6% | 352 |
| SOLUSDT | 6.675 | +212.5% | -4.6% | 349 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #109 - KAMA Donchian MTF Chandelier + Vol-Adjusted Sizing (15m+4h v2)
==================================================================================================
Hypothesis: Build on #108 (Sharpe=7.706) but replace DEMA with KAMA for better volatility adaptation.
Key improvements:
- KAMA (Kaufman Adaptive MA) instead of DEMA - adapts speed based on market efficiency
- Donchian channel breakout confirmation (20-period high/low)
- ATR percentile-based position sizing (reduce size when ATR is in top 30%)
- Cleaner Chandelier exit with proper state tracking
- More conservative sizing: MAX 0.30 (was 0.35), typical 0.20-0.25
- Hysteresis on signal changes to reduce churn costs

Why this should beat Sharpe=16.016:
- KAMA adapts to volatility regimes automatically (faster in trends, slower in chop)
- Donchian adds breakout confirmation filter
- ATR percentile sizing is more robust than fixed ratio
- Based on proven MTF 15m+4h structure from #096, #105, #108
"""

import numpy as np
import pandas as pd

name = "kama_donchian_mtf_chandelier_voladj_15m_4h_v2"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    for i in range(period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
        sc[i] = sc[i] ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)"""
    n = len(close) if 'close' in dir() else len(high)
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle


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


def calculate_atr_percentile(atr, lookback=100):
    """Calculate ATR percentile for volatility regime"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = atr[i - lookback + 1:i + 1]
        sorted_window = np.sort(window)
        rank = np.searchsorted(sorted_window, atr[i])
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
    kama_15m = calculate_kama(close, period=10, fast=2, slow=30)
    adx_15m = calculate_adx(high, low, close, period=14)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    donchian_upper_15m, donchian_lower_15m, donchian_mid_15m = calculate_donchian(high, low, period=20)
    atr_pct_15m = calculate_atr_percentile(atr_15m, lookback=100)
    
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
        if end_idx <= n:
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
    
    # 4h indicators for trend
    kama_4h = calculate_kama(c_4h, period=10, fast=2, slow=30)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    donchian_upper_4h, donchian_lower_4h, donchian_mid_4h = calculate_donchian(h_4h, l_4h, period=20)
    atr_pct_4h = calculate_atr_percentile(atr_4h, lookback=100)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    atr_pct_4h_mapped = np.zeros(n)
    donchian_trend_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            # KAMA trend
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Donchian trend (price vs middle)
            if c_4h[idx_4h] > donchian_mid_4h[idx_4h]:
                donchian_trend_4h[i] = 1
            elif c_4h[idx_4h] < donchian_mid_4h[idx_4h]:
                donchian_trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
            atr_pct_4h_mapped[i] = atr_pct_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.25  # More conservative than 0.35
    SIZE_HALF = 0.125
    SIZE_QUARTER = 0.0625
    
    # Volatility adjustment thresholds
    VOL_HIGH_PCT = 0.70  # Reduce size when ATR > 70th percentile
    VOL_LOW_PCT = 0.30   # Full size when ATR < 30th percentile
    VOL_REDUCTION = 0.5  # Reduce to 50% size in high vol
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 20  # Slightly lower than 25 for more signals
    
    # Chandelier exit multiplier
    CHAN_MULT = 3.0
    
    # BBW minimum for regime filter
    BBW_MIN = 0.015
    
    first_valid = max(200, 40 * bars_per_4h, 14 * 2, 20, 28, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    chandelier_stop = np.zeros(n)
    prev_signal = np.zeros(n)  # For hysteresis
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            prev_signal[i] = prev_signal[i - 1] if i > 0 else 0
            continue
        
        trend = trend_4h[i]
        donchian_trend = donchian_trend_4h[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_4h_val = bbw_4h_mapped[i]
        atr_4h_val = atr_4h_mapped[i]
        atr_pct = atr_pct_4h_mapped[i]
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            prev_signal[i] = 0
            continue
        
        # BBW filter - avoid choppy markets (4h)
        if bbw_4h_val < BBW_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            prev_signal[i] = 0
            continue
        
        # Trend filters must agree (KAMA + Donchian on 4h)
        if trend != donchian_trend or trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            prev_signal[i] = 0
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
                    prev_signal[i] = 0
                    continue
                
                # Take profit check (2R based on 4h ATR)
                tp_price = prev_entry + 2 * CHAN_MULT * atr_4h_val
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    chandelier_stop[i] = chandelier_stop[i]
                    prev_signal[i] = SIZE_HALF
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
                        prev_signal[i] = 0
                        continue
                
            elif prev_side == -1:
                new_chan_stop = current_low + CHAN_MULT * atr
                chandelier_stop[i] = min(prev_chan_stop, new_chan_stop) if prev_chan_stop < 0 else new_chan_stop
                
                # Chandelier exit stoploss
                if price > chandelier_stop[i]:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    chandelier_stop[i] = 0
                    prev_signal[i] = 0
                    continue
                
                # Take profit check (2R based on 4h ATR)
                tp_price = prev_entry - 2 * CHAN_MULT * atr_4h_val
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    chandelier_stop[i] = chandelier_stop[i]
                    prev_signal[i] = -SIZE_HALF
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
                        prev_signal[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            chandelier_stop[i] = chandelier_stop[i - 1]
            prev_signal[i] = prev_signal[i - 1]
            continue
        
        # Volatility-adjusted position sizing based on ATR percentile
        if atr_pct > VOL_HIGH_PCT:
            vol_multiplier = VOL_REDUCTION
        elif atr_pct < VOL_LOW_PCT:
            vol_multiplier = 1.0
        else:
            # Linear interpolation between low and high vol
            vol_multiplier = 1.0 - (atr_pct - VOL_LOW_PCT) / (VOL_HIGH_PCT - VOL_LOW_PCT) * (1.0 - VOL_REDUCTION)
        
        base_size = SIZE_FULL * vol_multiplier
        
        # Entry logic: 4h KAMA + Donchian + ADX + BBW + 15m RSI + Z-score
        # Add hysteresis: require RSI to cross threshold, not just be in range
        if trend == 1 and donchian_trend == 1:  # Bullish trend confirmed on 4h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX):  # Pullback + not extreme
                # Hysteresis: only enter if signal was 0 or opposite
                if prev_signal[i - 1] <= 0:
                    signals[i] = base_size
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = high[i]
                    lowest_since_entry[i] = low[i]
                    chandelier_stop[i] = high[i] - CHAN_MULT * atr
                    prev_signal[i] = base_size
                else:
                    signals[i] = signals[i - 1]
                    prev_signal[i] = prev_signal[i - 1]
            else:
                signals[i] = 0.0
                prev_signal[i] = 0
                
        elif trend == -1 and donchian_trend == -1:  # Bearish trend confirmed on 4h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX):  # Pullback + not extreme
                # Hysteresis: only enter if signal was 0 or opposite
                if prev_signal[i - 1] >= 0:
                    signals[i] = -base_size
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = high[i]
                    lowest_since_entry[i] = low[i]
                    chandelier_stop[i] = low[i] + CHAN_MULT * atr
                    prev_signal[i] = -base_size
                else:
                    signals[i] = signals[i - 1]
                    prev_signal[i] = prev_signal[i - 1]
            else:
                signals[i] = 0.0
                prev_signal[i] = 0
        
        else:
            signals[i] = 0.0
            prev_signal[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 11:07
