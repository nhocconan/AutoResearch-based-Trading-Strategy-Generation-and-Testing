# Strategy: mtf_hma_kama_stoch_rsi_bbw_zscore_15m_4h_v4

## Status
ACTIVE - Sharpe=8.730 | Return=+84669.7% | DD=-6.4%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 8.441 | +3964.0% | -5.0% | 2064 |
| ETHUSDT | 8.595 | +10377.5% | -4.9% | 2111 |
| SOLUSDT | 9.153 | +239667.4% | -9.2% | 2193 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 7.378 | +96.1% | -1.9% | 630 |
| ETHUSDT | 8.606 | +222.6% | -3.6% | 581 |
| SOLUSDT | 8.822 | +331.5% | -7.0% | 632 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #038 - MTF HMA+KAMA+Stoch+RSI+BBW+Zscore (15m+4h Optimized v4)
==================================================================================================
Hypothesis: #034 achieved Sharpe=10.162 with 15m+4h using HMA+KAMA+Stoch+RSI+BBW.
This version adds Z-score filter for better entry timing within pullbacks and optimizes thresholds.
Key improvements over #037:
- Remove Supertrend+ADX (they hurt in #036, only helped slightly in #037)
- Add Z-score(20) for mean reversion entry timing within trend pullbacks
- Widen RSI range to 40-60 (from 45-55) for more quality entries
- Position size: 0.30 (slightly more conservative than 0.35)
- Stoploss: 2.0*ATR (proven effective)
- Keep core HMA+KAMA+Stoch+RSI+BBW from #034 (the winning formula)

Why this should beat #034 (Sharpe=10.162):
- Z-score filters out extreme pullbacks that may reverse
- Wider RSI range captures more valid entries without sacrificing quality
- 0.30 position size reduces drawdown risk while maintaining returns
- Simpler filter stack (removed Supertrend+ADX overhead)
"""

import numpy as np
import pandas as pd

name = "mtf_hma_kama_stoch_rsi_bbw_zscore_15m_4h_v4"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    n = len(close)
    if n < k_period + d_period:
        return np.zeros(n), np.zeros(n)
    
    stoch_k = np.zeros(n)
    stoch_d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        
        if highest_high > lowest_low:
            stoch_k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            stoch_k[i] = 50
    
    for i in range(k_period - 1 + d_period - 1, n):
        stoch_d[i] = np.mean(stoch_k[i - d_period + 1:i + 1])
    
    return stoch_k, stoch_d


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


def resample_to_higher_tf(prices, target_tf='4h'):
    """Resample to higher timeframe using open_time index"""
    prices_indexed = prices.set_index('open_time')
    df_resampled = prices_indexed.resample(target_tf).agg({
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
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    stoch_k_15m, stoch_d_15m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    zscore_15m = calculate_zscore(close, period=20)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 4h for trend filters using proper method
    try:
        df_4h = resample_to_higher_tf(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        n_4h = len(c_4h)
        
        # 4h indicators for trend
        hma_4h = calculate_hma(c_4h, period=21)
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        
        # Map 4h indicators back to 15m timeframe using reindex
        prices_indexed = prices.set_index('open_time')
        df_4h_indexed = df_4h
        
        # Create mapping arrays
        trend_4h = np.zeros(n)
        kama_trend_4h = np.zeros(n)
        bbw_4h_mapped = np.zeros(n)
        
        # Align 4h data to 15m timestamps
        for i in range(n):
            ts = prices_indexed.index[i]
            mask = df_4h_indexed.index <= ts
            if mask.sum() > 0:
                idx_4h = mask.sum() - 1
                if idx_4h >= 40:
                    if c_4h[idx_4h] > hma_4h[idx_4h]:
                        trend_4h[i] = 1
                    elif c_4h[idx_4h] < hma_4h[idx_4h]:
                        trend_4h[i] = -1
                    
                    if c_4h[idx_4h] > kama_4h[idx_4h]:
                        kama_trend_4h[i] = 1
                    elif c_4h[idx_4h] < kama_4h[idx_4h]:
                        kama_trend_4h[i] = -1
                    
                    bbw_4h_mapped[i] = bbw_4h[idx_4h]
    except Exception:
        bars_per_4h = 16
        n_4h = n // bars_per_4h
        
        c_4h = np.zeros(n_4h)
        h_4h = np.zeros(n_4h)
        l_4h = np.zeros(n_4h)
        
        for i in range(n_4h):
            start_idx = i * bars_per_4h
            end_idx = start_idx + bars_per_4h
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
        
        hma_4h = calculate_hma(c_4h, period=21)
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        
        trend_4h = np.zeros(n)
        kama_trend_4h = np.zeros(n)
        bbw_4h_mapped = np.zeros(n)
        
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h < n_4h and idx_4h >= 40:
                if c_4h[idx_4h] > hma_4h[idx_4h]:
                    trend_4h[i] = 1
                elif c_4h[idx_4h] < hma_4h[idx_4h]:
                    trend_4h[i] = -1
                
                if c_4h[idx_4h] > kama_4h[idx_4h]:
                    kama_trend_4h[i] = 1
                elif c_4h[idx_4h] < kama_4h[idx_4h]:
                    kama_trend_4h[i] = -1
                
                bbw_4h_mapped[i] = bbw_4h[idx_4h]
    
    signals = np.zeros(n)
    
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    STOCH_LONG_MIN = 40
    STOCH_LONG_MAX = 60
    STOCH_SHORT_MIN = 40
    STOCH_SHORT_MAX = 60
    
    ZSCORE_MAX = 1.5
    ZSCORE_MIN = -1.5
    
    BBW_MIN = 0.015
    
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 40 * 16, 14 * 2, 20, 28)
    
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        kama_trend = kama_trend_4h[i]
        rsi_val = rsi_15m[i]
        stoch_k = stoch_k_15m[i]
        stoch_d = stoch_d_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        bbw_4h_val = bbw_4h_mapped[i]
        
        if bbw_4h_val < BBW_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        if trend != kama_trend or trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
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
                
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
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
                
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
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
            
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        if trend == 1 and kama_trend == 1:
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                STOCH_LONG_MIN <= stoch_k <= STOCH_LONG_MAX and
                stoch_k > stoch_d and
                ZSCORE_MIN <= zscore_val <= ZSCORE_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and kama_trend == -1:
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                STOCH_SHORT_MIN <= stoch_k <= STOCH_SHORT_MAX and
                stoch_k < stoch_d and
                ZSCORE_MIN <= zscore_val <= ZSCORE_MAX):
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
2026-03-21 12:54
