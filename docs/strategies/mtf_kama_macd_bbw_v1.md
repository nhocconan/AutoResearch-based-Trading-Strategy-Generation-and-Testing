# Strategy: mtf_kama_macd_bbw_v1

## Status
ACTIVE - Sharpe=0.123 | Return=+41.8% | DD=-19.6%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.395 | +5.2% | -15.3% | 645 |
| ETHUSDT | -0.127 | +13.3% | -25.5% | 609 |
| SOLUSDT | 0.890 | +107.0% | -17.9% | 588 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 1.441 | +17.1% | -2.5% | 194 |
| ETHUSDT | 1.932 | +30.1% | -3.2% | 180 |
| SOLUSDT | 0.375 | +10.7% | -6.8% | 179 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #014 - KAMA Adaptive Trend + MACD Histogram Momentum + BBW Regime Filter
====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better 
than EMA/HMA, providing cleaner trend signals during choppy periods. Combined with 
MACD histogram for momentum entry timing (different from RSI pullback in #005/#007) 
and Bollinger Band Width for regime detection (proven in #007 with Sharpe=4.711).

Key differences from current best (#005 EMA+RSI+Z-score):
- KAMA(10,2,30) trend filter instead of EMA - adapts to volatility automatically
- MACD(12,26,9) histogram cross for entry (momentum-based, not mean-reversion like RSI)
- BBW percentile filter instead of Z-score - detects volatility regime (squeeze vs expansion)
- 4h KAMA trend + 1h MACD entries (proven MTF structure from #005)
- Trailing stoploss at 2*ATR, take profit at 2R (reduce to half)
- Discrete signal levels: 0.0, ±0.20, ±0.30 to minimize churn costs

Why this might beat Sharpe=5.525:
- KAMA reduces whipsaws during choppy markets better than fixed EMA
- MACD histogram captures momentum shifts earlier than RSI pullback
- BBW filter avoids trading during low-volatility squeezes (major improvement over Z-score)
- Multi-timeframe structure proven to 2x Sharpe vs single timeframe
- Different signal combination than #005/#007 - may capture different market regimes
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_bbw_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency ratio (ER)
    High ER (trending) = fast smoothing, Low ER (choppy) = slow smoothing
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < er_period + fast_sc:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = change / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    sc = np.zeros(n)
    fast_constant = 2.0 / (fast_sc + 1)
    slow_constant = 2.0 / (slow_sc + 1)
    
    for i in range(er_period, n):
        sc[i] = er[i] * (fast_constant - slow_constant) + slow_constant
        sc[i] = sc[i] ** 2  # Square for smoother adaptation
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns: MACD line, Signal line, Histogram
    """
    n = len(close)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    if n < slow + signal:
        return macd_line, signal_line, histogram
    
    # Calculate EMAs
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    # MACD Line
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal Line (EMA of MACD)
    valid_macd_start = slow - 1
    macd_values = macd_line[valid_macd_start:valid_macd_start + signal]
    signal_line[valid_macd_start + signal - 1] = np.mean(macd_values)
    
    for i in range(valid_macd_start + signal, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    # Histogram
    for i in range(valid_macd_start + signal - 1, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """
    Calculate Bollinger Bands
    Returns: upper, middle, lower, bandwidth
    """
    n = len(close)
    upper = np.zeros(n)
    middle = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    
    for i in range(period - 1, n):
        middle[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bandwidth[i] = 0
    
    return upper, middle, lower, bandwidth


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    macd_1h, signal_1h, histogram_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper_1h, bb_mid_1h, bb_lower_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # 4h KAMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    n_4h = len(c_4h)
    
    # Calculate 4h KAMA for trend
    kama_4h = calculate_kama(c_4h, er_period=10, fast_sc=2, slow_sc=30)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(n_4h)
    kama_period = 40  # Need enough data for KAMA
    
    for i in range(kama_period, n_4h):
        # Price above KAMA and KAMA sloping up = bullish
        price_above_kama = c_4h[i] > kama_4h[i]
        kama_sloping_up = kama_4h[i] > kama_4h[i - 5] if i >= 5 else False
        
        # Price below KAMA and KAMA sloping down = bearish
        price_below_kama = c_4h[i] < kama_4h[i]
        kama_sloping_down = kama_4h[i] < kama_4h[i - 5] if i >= 5 else False
        
        if price_above_kama and kama_sloping_up:
            trend_4h[i] = 1  # Bullish
        elif price_below_kama and kama_sloping_down:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Calculate BBW percentile for regime filter (rolling 100-period)
    bbw_percentile = np.zeros(n)
    bbw_lookback = 100
    
    for i in range(bbw_lookback - 1, n):
        bbw_window = bbw_1h[i - bbw_lookback + 1:i + 1]
        bbw_percentile[i] = np.sum(bbw_window <= bbw_1h[i]) / bbw_lookback * 100
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.30   # Full position (conservative)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # MACD histogram thresholds for momentum entry
    MACD_HIST_THRESHOLD = 0.0  # Cross above/below zero
    MACD_HIST_MIN = 0.0001     # Minimum histogram magnitude to confirm
    
    # BBW percentile thresholds for regime filter
    BBW_MIN_PERCENTILE = 30    # Avoid trading during extreme squeeze (< 30th percentile)
    BBW_MAX_PERCENTILE = 85    # Caution during extreme expansion (> 85th percentile)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(100, 40, 40, 100)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    initial_stop = np.zeros(n)  # Track initial stoploss level
    
    for i in range(first_valid, n):
        if np.isnan(macd_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bbw_1h[i]) or np.isnan(bbw_percentile[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        hist_val = histogram_1h[i]
        prev_hist_val = histogram_1h[i - 1] if i > 0 else 0
        bbw_pct = bbw_percentile[i]
        atr = atr_1h[i]
        price = close[i]
        
        # BBW regime filter - avoid extreme squeeze (low volatility = false breakouts)
        if bbw_pct < BBW_MIN_PERCENTILE:
            if i > 0 and position_side[i - 1] != 0:
                # Hold existing position but don't add
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                if position_side[i-1] == 1:
                    highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                    lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                elif position_side[i-1] == -1:
                    highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                    lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                initial_stop[i] = initial_stop[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_stop = initial_stop[i - 1] if initial_stop[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                
                # Trailing stoploss (2*ATR from entry, or trail from highest)
                trail_stop = highest_since_entry[i] - ATR_STOP_MULT * atr if highest_since_entry[i] > 0 else prev_entry - ATR_STOP_MULT * atr
                stoploss_price = max(prev_entry - ATR_STOP_MULT * atr, trail_stop)
                
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
                
                # MACD histogram exit for longs (momentum fading)
                if prev_hist_val > 0 and hist_val < 0:  # Histogram crossed below zero
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                    
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                
                # Trailing stoploss
                trail_stop = lowest_since_entry[i] + ATR_STOP_MULT * atr if lowest_since_entry[i] > 0 else prev_entry + ATR_STOP_MULT * atr
                stoploss_price = min(prev_entry + ATR_STOP_MULT * atr, trail_stop)
                
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
                
                # MACD histogram exit for shorts (momentum fading)
                if prev_hist_val < 0 and hist_val > 0:  # Histogram crossed above zero
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
        
        # Entry logic with MACD histogram momentum confirmation
        position_size = SIZE_FULL
        
        if trend == 1:  # 4h uptrend + BBW OK
            # MACD histogram crosses above zero (momentum turning positive)
            if prev_hist_val <= MACD_HIST_THRESHOLD and hist_val > MACD_HIST_THRESHOLD + MACD_HIST_MIN:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                initial_stop[i] = price - ATR_STOP_MULT * atr
            else:
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = initial_stop[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    
        elif trend == -1:  # 4h downtrend + BBW OK
            # MACD histogram crosses below zero (momentum turning negative)
            if prev_hist_val >= MACD_HIST_THRESHOLD and hist_val < MACD_HIST_THRESHOLD - MACD_HIST_MIN:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                initial_stop[i] = price + ATR_STOP_MULT * atr
            else:
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = initial_stop[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            initial_stop[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 08:53
