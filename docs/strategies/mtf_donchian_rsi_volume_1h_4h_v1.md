# Strategy: mtf_donchian_rsi_volume_1h_4h_v1

## Status
ACTIVE - Sharpe=0.784 | Return=+63.0% | DD=-7.2%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 0.491 | +36.0% | -5.5% | 1191 |
| ETHUSDT | 0.608 | +45.4% | -7.1% | 1058 |
| SOLUSDT | 1.255 | +107.7% | -9.0% | 961 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 2.198 | +23.8% | -3.5% | 320 |
| ETHUSDT | 1.869 | +28.0% | -4.0% | 306 |
| SOLUSDT | 2.790 | +49.8% | -8.7% | 300 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #004 - MTF Donchian+RSI+Volume (1h+4h)
==================================================================================================
Hypothesis: 1h timeframe with Donchian channel trend + RSI pullback + Volume filter will reduce
noise and improve risk-adjusted returns compared to 15m strategies. The 1h timeframe should have
fewer false signals while still capturing meaningful moves. Donchian breakouts provide clear
trend direction, RSI pullbacks give good entry timing, and volume confirms genuine moves.

Key differences from #038:
- Timeframe: 1h instead of 15m (4x fewer bars, less fee churn)
- Trend: Donchian(20) breakout instead of HMA/KAMA
- Entry: RSI(14) pullback to 40-60 zone
- Filter: Volume spike (>1.5x average)
- Position size: 0.35 (slightly higher since fewer trades)
- Stoploss: 2.5*ATR (wider for 1h timeframe)
- Add 4h ADX filter to ensure strong trend environment
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_rsi_volume_1h_4h_v1"
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


def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


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
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    tr_smooth[period - 1] = np.sum(tr[1:period])
    plus_smooth = np.sum(plus_dm[1:period])
    minus_smooth = np.sum(minus_dm[1:period])
    
    for i in range(period, n):
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
        plus_smooth = plus_smooth - plus_smooth / period + plus_dm[i]
        minus_smooth = minus_smooth - minus_smooth / period + minus_dm[i]
        
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_smooth / tr_smooth[i]
            minus_di[i] = 100 * minus_smooth / tr_smooth[i]
    
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


def calculate_volume_sma(volume, period=20):
    """Calculate Volume SMA"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    volume_sma = np.zeros(n)
    for i in range(period - 1, n):
        volume_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return volume_sma


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
    volume = prices["volume"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    donchian_upper_1h, donchian_lower_1h = calculate_donchian_channels(high, low, period=20)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Resample to 4h for trend filters using proper method
    try:
        df_4h = resample_to_higher_tf(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        v_4h = df_4h['volume'].values
        n_4h = len(c_4h)
        
        # 4h indicators for trend
        donchian_upper_4h, donchian_lower_4h = calculate_donchian_channels(h_4h, l_4h, period=20)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        # Map 4h indicators back to 1h timeframe using reindex
        prices_indexed = prices.set_index('open_time')
        df_4h_indexed = df_4h
        
        # Create mapping arrays
        trend_4h = np.zeros(n)
        adx_4h_mapped = np.zeros(n)
        donchian_mid_4h = np.zeros(n)
        
        # Align 4h data to 1h timestamps with shift(1) to avoid look-ahead
        df_4h_shifted = df_4h_indexed.shift(1)
        
        for i in range(n):
            ts = prices_indexed.index[i]
            # Find the most recent completed 4h bar
            mask = df_4h_shifted.index <= ts
            if mask.sum() > 0:
                idx_4h = mask.sum() - 1
                if idx_4h >= 20:
                    # Trend: price above Donchian mid = bullish, below = bearish
                    mid_4h = (donchian_upper_4h[idx_4h] + donchian_lower_4h[idx_4h]) / 2
                    donchian_mid_4h[i] = mid_4h
                    
                    if c_4h[idx_4h] > mid_4h:
                        trend_4h[i] = 1
                    elif c_4h[idx_4h] < mid_4h:
                        trend_4h[i] = -1
                    
                    adx_4h_mapped[i] = adx_4h[idx_4h]
    except Exception:
        # Fallback: simple bar counting
        bars_per_4h = 4
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
        
        donchian_upper_4h, donchian_lower_4h = calculate_donchian_channels(h_4h, l_4h, period=20)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        trend_4h = np.zeros(n)
        adx_4h_mapped = np.zeros(n)
        donchian_mid_4h = np.zeros(n)
        
        for i in range(n):
            idx_4h = max(0, i // bars_per_4h - 1)  # Shift by 1 to avoid look-ahead
            if idx_4h < n_4h and idx_4h >= 20:
                mid_4h = (donchian_upper_4h[idx_4h] + donchian_lower_4h[idx_4h]) / 2
                donchian_mid_4h[i] = mid_4h
                
                if c_4h[idx_4h] > mid_4h:
                    trend_4h[i] = 1
                elif c_4h[idx_4h] < mid_4h:
                    trend_4h[i] = -1
                
                adx_4h_mapped[i] = adx_4h[idx_4h]
    
    signals = np.zeros(n)
    
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    VOLUME_MULT = 1.5
    ADX_MIN = 20
    
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 20 * 4, 14 * 2, 20, 28)
    
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        adx_val = adx_4h_mapped[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        vol = volume[i]
        vol_avg = volume_sma_1h[i]
        
        # ADX filter: only trade when trend is strong
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Volume filter: only trade on above-average volume
        if vol_avg > 0 and vol < vol_avg * VOLUME_MULT:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Manage existing positions
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
        
        # New entry logic
        if trend == 1:
            # Long: RSI pullback in bullish trend
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1:
            # Short: RSI pullback in bearish trend
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
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
2026-03-21 13:04
