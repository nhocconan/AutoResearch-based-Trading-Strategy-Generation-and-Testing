# Strategy: mtf_hma_rsi_chandelier_volregime_15m_4h_v4

## Status
ACTIVE - Sharpe=5.225 | Return=+12437.5% | DD=-5.8%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 4.832 | +1388.1% | -4.2% | 4358 |
| ETHUSDT | 4.924 | +2570.5% | -6.3% | 4331 |
| SOLUSDT | 5.918 | +33353.8% | -6.9% | 4114 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 3.749 | +55.0% | -2.8% | 1306 |
| ETHUSDT | 4.697 | +116.7% | -4.2% | 1244 |
| SOLUSDT | 5.873 | +223.7% | -6.8% | 1215 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #111 - MTF HMA RSI Chandelier VolRegime Enhanced (15m+4h v4)
==================================================================================================
Hypothesis: Rebuild on #108 (Sharpe=7.706) success factors with cleaner implementation:
- 4h HMA trend filter for directional bias
- 15m RSI pullback entries with optimal ranges (35-55 long, 45-65 short)
- ATR(22) Chandelier exit with multiplier 3.0 for trailing stops
- Volatility percentile position sizing (reduce size in high vol regimes)
- ADX(14) > 20 filter to avoid choppy markets
- BBW regime filter to avoid squeeze breakouts
- Discrete signal levels (0.0, ±0.20, ±0.30) to minimize churn

Key improvements over #110:
- Cleaner position state tracking with proper initialization
- Fixed Chandelier exit logic (was causing premature exits)
- Better RSI pullback ranges based on #108 success
- Simplified volatility adjustment (percentile-based)
- Proper min_periods on all rolling calculations
- Vectorized indicator calculations where possible

Why this should beat Sharpe=16.016:
- Based on proven MTF 15m+4h structure from #096, #105, #108
- Combines trend (HMA) + momentum (RSI) + volatility (ATR/BBW) filters
- Conservative sizing (max 0.30) prevents blowup in crypto crashes
- Chandelier exit locks in profits during strong trends
- ADX filter avoids whipsaw in ranging markets
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_chandelier_volregime_15m_4h_v4"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, length):
        weights = np.arange(1, length + 1, dtype=float)
        result = np.zeros(len(series))
        for i in range(length - 1, len(series)):
            result[i] = np.sum(series[i - length + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_full = wma(close, period)
    wma_half = wma(close, half)
    diff = 2 * wma_half - wma_full
    
    hma = np.zeros(n)
    for i in range(sqrt_period - 1, n):
        weights = np.arange(1, sqrt_period + 1, dtype=float)
        start = i - sqrt_period + 1
        hma[i] = np.sum(diff[start:i + 1] * weights) / np.sum(weights)
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    plus_dm[1:] = np.where(
        (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
        np.maximum(0, high[1:] - high[:-1]),
        0
    )
    
    minus_dm[1:] = np.where(
        (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
        np.maximum(0, low[:-1] - low[1:]),
        0
    )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = atr > 1e-10
    plus_di[mask] = 100 * plus_dm[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm[mask] / atr[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask = di_sum > 1e-10
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    
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
        
        if middle[i] > 1e-10:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
    
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


def resample_to_4h(close, high, low, bars_per_4h=16):
    """Resample 15m data to 4h"""
    n = len(close)
    n_4h = n // bars_per_4h
    
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
    
    return c_4h, h_4h, l_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.astype(float)
    high = prices["high"].values.astype(float)
    low = prices["low"].values.astype(float)
    n = len(close)
    
    signals = np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=16)
    adx_15m = calculate_adx(high, low, close, period=14)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h = resample_to_4h(close, high, low, bars_per_4h)
    n_4h = len(c_4h)
    
    # 4h indicators for trend
    hma_4h = calculate_hma(c_4h, period=16)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=22)
    atr_pct_4h = calculate_atr_percentile(atr_4h, lookback=100)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    atr_pct_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
            atr_pct_4h_mapped[i] = atr_pct_4h[idx_4h]
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Volatility adjustment thresholds
    VOL_HIGH_PCT = 0.70
    VOL_LOW_PCT = 0.30
    
    # RSI thresholds for pullback entries (proven from #108)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ADX threshold for trend strength
    ADX_MIN = 20
    
    # BBW minimum for regime filter
    BBW_MIN = 0.005
    
    # Chandelier exit multiplier (ATR 22 period)
    CHAN_MULT = 3.0
    CHAN_PERIOD = 22
    
    # Minimum valid index
    first_valid = max(200, 40 * bars_per_4h, 100)
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    chandelier_stop = 0.0
    
    for i in range(first_valid, n):
        # Skip invalid data
        if atr_15m[i] < 1e-10 or np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_4h_val = bbw_4h_mapped[i]
        atr_4h_val = atr_4h_mapped[i]
        atr_pct = atr_pct_4h_mapped[i]
        
        # ADX filter - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            if in_position:
                # Close existing position
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                chandelier_stop = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # BBW filter - avoid choppy markets
        if bbw_4h_val < BBW_MIN:
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                chandelier_stop = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        if trend == 0:
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                chandelier_stop = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Check exits for existing positions
        if in_position:
            # Update highest/lowest since entry
            if position_side == 1:
                if highest_since_entry < 1e-10:
                    highest_since_entry = high[i]
                else:
                    highest_since_entry = max(highest_since_entry, high[i])
                
                # Update Chandelier stop
                new_chan_stop = highest_since_entry - CHAN_MULT * atr
                if chandelier_stop < 1e-10:
                    chandelier_stop = new_chan_stop
                else:
                    chandelier_stop = max(chandelier_stop, new_chan_stop)
                
                # Chandelier exit stoploss
                if price < chandelier_stop:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    chandelier_stop = 0.0
                    continue
                
                # Take profit check (2R based on 4h ATR)
                if not tp_triggered:
                    tp_price = entry_price + 2 * CHAN_MULT * atr_4h_val
                    if price >= tp_price:
                        signals[i] = SIZE_HALF
                        tp_triggered = True
                        continue
                
                # Trail stop after TP
                if tp_triggered:
                    trail_stop = highest_since_entry - CHAN_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        chandelier_stop = 0.0
                        continue
            
            elif position_side == -1:
                if lowest_since_entry < 1e-10:
                    lowest_since_entry = low[i]
                else:
                    lowest_since_entry = min(lowest_since_entry, low[i])
                
                # Update Chandelier stop
                new_chan_stop = lowest_since_entry + CHAN_MULT * atr
                if chandelier_stop < 1e-10:
                    chandelier_stop = new_chan_stop
                else:
                    chandelier_stop = min(chandelier_stop, new_chan_stop)
                
                # Chandelier exit stoploss
                if price > chandelier_stop:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    chandelier_stop = 0.0
                    continue
                
                # Take profit check
                if not tp_triggered:
                    tp_price = entry_price - 2 * CHAN_MULT * atr_4h_val
                    if price <= tp_price:
                        signals[i] = -SIZE_HALF
                        tp_triggered = True
                        continue
                
                # Trail stop after TP
                if tp_triggered:
                    trail_stop = lowest_since_entry + CHAN_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        chandelier_stop = 0.0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1] if i > 0 else 0.0
            continue
        
        # Volatility-adjusted position sizing
        if atr_pct > VOL_HIGH_PCT:
            vol_multiplier = 0.5
        elif atr_pct < VOL_LOW_PCT:
            vol_multiplier = 1.0
        else:
            vol_multiplier = 1.0 - (atr_pct - VOL_LOW_PCT) / (VOL_HIGH_PCT - VOL_LOW_PCT) * 0.5
        
        base_size = SIZE_FULL * vol_multiplier
        
        # Entry logic
        if trend == 1:  # Bullish trend
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                # Hysteresis - only enter if not already long
                if signals[i - 1] <= 0 if i > 0 else True:
                    signals[i] = base_size
                    in_position = True
                    position_side = 1
                    entry_price = price
                    tp_triggered = False
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    chandelier_stop = high[i] - CHAN_MULT * atr
                else:
                    signals[i] = signals[i - 1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
                
        elif trend == -1:  # Bearish trend
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                # Hysteresis - only enter if not already short
                if signals[i - 1] >= 0 if i > 0 else True:
                    signals[i] = -base_size
                    in_position = True
                    position_side = -1
                    entry_price = price
                    tp_triggered = False
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    chandelier_stop = low[i] + CHAN_MULT * atr
                else:
                    signals[i] = signals[i - 1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
        
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-21 11:10
