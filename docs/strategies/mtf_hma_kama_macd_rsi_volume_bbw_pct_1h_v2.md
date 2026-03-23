# Strategy: mtf_hma_kama_macd_rsi_volume_bbw_pct_1h_v2

## Status
ACTIVE - Sharpe=2.727 | Return=+310.4% | DD=-7.0%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 2.155 | +112.1% | -4.6% | 471 |
| ETHUSDT | 2.298 | +149.0% | -7.7% | 471 |
| SOLUSDT | 3.727 | +670.1% | -8.7% | 488 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 3.368 | +26.4% | -2.5% | 130 |
| ETHUSDT | 2.548 | +30.6% | -6.5% | 141 |
| SOLUSDT | 4.361 | +73.4% | -7.1% | 143 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #022 - MTF HMA+KAMA+MACD+RSI+Volume+BBW-Percentile (1h+4h v2)
==================================================================================================
Hypothesis: Building on #021 (Sharpe=4.629) which proved 15m+1h MTF works excellently.
This experiment tests 1h+4h MTF combination with refined parameters:

New improvements vs #021:
- Timeframe: 1h entries + 4h trend (slower, fewer false signals, lower fees)
- Add MACD(12,26,9) histogram for momentum confirmation at entry
- Position size: 0.25 (more conservative than 0.30 to reduce drawdown)
- RSI range: 40-60 for longs, 40-60 for shorts (symmetric pullback detection)
- MACD histogram must align with trend direction
- Volume ratio filter: 1.3x average (higher than 1.2x for better confirmation)
- ATR stoploss: 2.2*ATR (slightly tighter than 2.0*ATR)

Why this should beat #021:
- 4h trend is more stable than 1h trend (fewer whipsaws)
- MACD adds momentum confirmation missing in #021
- Lower position size (0.25 vs 0.30) reduces drawdown risk
- Fewer trades = lower fee drag (1h vs 15m entries)
- Based on proven MTF structure from #012, #019, #021
"""

import numpy as np
import pandas as pd

name = "mtf_hma_kama_macd_rsi_volume_bbw_pct_1h_v2"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    # Calculate EMA fast
    multiplier_fast = 2.0 / (fast + 1)
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = (close[i] - ema_fast[i - 1]) * multiplier_fast + ema_fast[i - 1]
    
    # Calculate EMA slow
    multiplier_slow = 2.0 / (slow + 1)
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = (close[i] - ema_slow[i - 1]) * multiplier_slow + ema_slow[i - 1]
    
    # MACD line
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal line (EMA of MACD)
    multiplier_signal = 2.0 / (signal + 1)
    first_signal_idx = slow - 1 + signal - 1
    if first_signal_idx < n:
        signal_line[first_signal_idx] = np.mean(macd_line[slow - 1:first_signal_idx + 1])
        for i in range(first_signal_idx + 1, n):
            signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier_signal + signal_line[i - 1]
    
    # Histogram
    for i in range(first_signal_idx, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


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


def calculate_volume_sma_ratio(volume, period=20):
    """Calculate volume ratio vs SMA"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    ratio = np.zeros(n)
    
    for i in range(period - 1, n):
        avg_volume = np.mean(volume[i - period + 1:i + 1])
        if avg_volume > 0:
            ratio[i] = volume[i] / avg_volume
        else:
            ratio[i] = 1.0
    
    return ratio


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile rank over lookback period"""
    n = len(bbw)
    if n < lookback:
        return np.zeros(n) * 50
    
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        current = bbw[i]
        rank = np.sum(window < current)
        percentile[i] = rank / lookback * 100
    
    return percentile


def resample_to_timeframe(prices, timeframe='4h'):
    """Resample prices to higher timeframe using open_time index"""
    prices_indexed = prices.set_index('open_time')
    
    df_resampled = prices_indexed.resample(timeframe).agg({
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
    volume = prices["volume"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    hma_1h = calculate_hma(close, period=21)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    macd_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    volume_ratio_1h = calculate_volume_sma_ratio(volume, period=20)
    _, _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
    
    # Resample to 4h for trend filters using proper method
    try:
        prices_indexed = prices.set_index('open_time')
        df_4h = prices_indexed.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate 4h indicators
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        hma_4h = calculate_hma(c_4h, period=21)
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
        
        # Map 4h indicators back to 1h timeframe using ffill
        trend_4h = np.zeros(len(c_4h))
        kama_trend_4h = np.zeros(len(c_4h))
        
        for i in range(len(c_4h)):
            if i >= 40:
                if c_4h[i] > hma_4h[i]:
                    trend_4h[i] = 1
                elif c_4h[i] < hma_4h[i]:
                    trend_4h[i] = -1
                
                if c_4h[i] > kama_4h[i]:
                    kama_trend_4h[i] = 1
                elif c_4h[i] < kama_4h[i]:
                    kama_trend_4h[i] = -1
        
        # Create 4h index for reindexing
        df_4h['trend'] = trend_4h
        df_4h['kama_trend'] = kama_trend_4h
        df_4h['bbw_pct'] = bbw_pct_4h
        
        # Reindex to 1h with ffill
        df_4h_reindexed = df_4h.reindex(prices_indexed.index, method='ffill')
        
        trend_4h_mapped = df_4h_reindexed['trend'].values
        kama_trend_4h_mapped = df_4h_reindexed['kama_trend'].values
        bbw_pct_4h_mapped = df_4h_reindexed['bbw_pct'].values
        
    except Exception:
        # Fallback: simple downsampling if resample fails
        bars_per_4h = 4
        n_4h = (n // bars_per_4h)
        
        c_4h = np.zeros(n_4h)
        for i in range(n_4h):
            start_idx = i * bars_per_4h
            end_idx = start_idx + bars_per_4h
            c_4h[i] = close[end_idx - 1]
        
        hma_4h = calculate_hma(c_4h, period=21)
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        
        trend_4h_mapped = np.zeros(n)
        kama_trend_4h_mapped = np.zeros(n)
        bbw_pct_4h_mapped = np.zeros(n)
        
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h < n_4h and idx_4h >= 40:
                if c_4h[idx_4h] > hma_4h[idx_4h]:
                    trend_4h_mapped[i] = 1
                elif c_4h[idx_4h] < hma_4h[idx_4h]:
                    trend_4h_mapped[i] = -1
                
                if c_4h[idx_4h] > kama_4h[idx_4h]:
                    kama_trend_4h_mapped[i] = 1
                elif c_4h[idx_4h] < kama_4h[idx_4h]:
                    kama_trend_4h_mapped[i] = -1
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.25
    SIZE_HALF = 0.125
    
    # RSI thresholds for pullback entries (symmetric range)
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # MACD histogram threshold
    MACD_MIN = 0.0  # Must be positive for longs, negative for shorts
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # Volume ratio threshold
    VOLUME_RATIO_MIN = 1.3
    
    # BBW percentile threshold (avoid bottom 20% = too choppy)
    BBW_PCT_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.2
    
    first_valid = max(200, 40 * 4, 14 * 2, 20, 28, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h_mapped[i]
        kama_trend = kama_trend_4h_mapped[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        macd_hist = macd_hist_1h[i]
        vol_ratio = volume_ratio_1h[i]
        bbw_pct = bbw_pct_4h_mapped[i]
        
        # BBW percentile filter - avoid choppy markets (bottom 20%)
        if bbw_pct < BBW_PCT_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (HMA + KAMA on 4h)
        if trend != kama_trend or trend == 0:
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
            
            # Stoploss check (2.2*ATR)
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
        
        # Entry logic: 4h HMA + KAMA + BBW% + 1h RSI + MACD + Volume + Z-score
        if trend == 1 and kama_trend == 1:  # Bullish trend confirmed on 4h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                macd_hist > MACD_MIN and
                abs(zscore_val) < ZSCORE_MAX and
                vol_ratio >= VOLUME_RATIO_MIN):  # Pullback + MACD positive + Volume confirm
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and kama_trend == -1:  # Bearish trend confirmed on 4h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                macd_hist < -MACD_MIN and
                abs(zscore_val) < ZSCORE_MAX and
                vol_ratio >= VOLUME_RATIO_MIN):  # Pullback + MACD negative + Volume confirm
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
2026-03-21 12:29
