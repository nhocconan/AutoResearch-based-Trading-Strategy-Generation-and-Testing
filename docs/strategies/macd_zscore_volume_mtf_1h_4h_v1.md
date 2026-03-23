# Strategy: macd_zscore_volume_mtf_1h_4h_v1

## Status
ACTIVE - Sharpe=0.149 | Return=+63.4% | DD=-24.8%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.304 | +11.4% | -13.1% | 47 |
| ETHUSDT | -0.428 | -1.5% | -27.5% | 81 |
| SOLUSDT | 1.180 | +180.1% | -33.9% | 3 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.847 | -6.8% | -9.9% | 58 |
| ETHUSDT | -0.830 | -4.5% | -14.3% | 63 |
| SOLUSDT | -0.404 | -0.1% | -14.0% | 65 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #016 - MACD Z-Score Momentum with 4h Trend Filter (1h Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.537) uses 4h primary + 1d trend. This uses 1h primary + 4h trend
with MACD momentum + Z-score filter. The baseline mtf_hma_rsi_zscore_v1 achieved Sharpe=5.4 using
4h trend + 1h entries + Z-score. This combines MACD histogram (proven in #007 Sharpe=0.488) with
Z-score mean reversion filter and volume confirmation.

Key innovations:
1. 1h PRIMARY + 4h HTF: More trade opportunities than 4h primary, but filtered by stronger 4h trend
2. MACD histogram momentum: Captures acceleration/deceleration better than RSI alone
3. Z-score(20) filter: Avoid entering when price is >2 std dev from mean (mean reversion risk)
4. Volume confirmation: Taker buy/sell ratio confirms institutional interest
5. Dynamic sizing: 0.15 base, 0.25 high conviction, 0.30 very high conviction
6. Tighter stoploss: 1.5 ATR (vs 2.0 in current best) for faster loss cutting

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- 1h timeframe captures more momentum moves than 4h
- MACD histogram is leading indicator (RSI is lagging)
- Z-score filter prevents buying tops/selling bottoms
- Volume confirmation reduces false breakouts
- Based on proven Sharpe=5.4 baseline architecture
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_zscore_volume_mtf_1h_4h_v1"
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
    """
    Hull Moving Average - faster than EMA, smoother than SMA
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        result = np.zeros(len(series))
        weights = np.arange(1, window + 1)
        for i in range(window - 1, len(series)):
            result[i] = np.sum(series[i - window + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    close_series = np.array(close)
    wma_half = wma(close_series, half)
    wma_full = wma(close_series, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    macd_line = (ema_fast - ema_slow).values
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score: (price - mean) / std"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    rolling_mean = close_series.rolling(window=period, min_periods=period).mean()
    rolling_std = close_series.rolling(window=period, min_periods=period).std()
    
    zscore = (close_series - rolling_mean) / rolling_std
    zscore = zscore.fillna(0).values
    
    return zscore


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    if n < len(atr) or len(atr) == 0:
        return np.zeros(n), np.zeros(n)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        if atr[i] == 0:
            continue
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    first_valid = np.where(atr > 0)[0]
    if len(first_valid) == 0:
        return supertrend, trend
    
    start_idx = first_valid[0]
    supertrend[start_idx] = upper_band[start_idx]
    trend[start_idx] = 1
    
    for i in range(start_idx + 1, n):
        if atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            trend[i] = trend[i - 1]
            continue
        
        if trend[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = max(supertrend[i - 1], lower_band[i])
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = min(supertrend[i - 1], upper_band[i])
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    zscore_1h = calculate_zscore(close, period=20)
    supertrend_1h, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    
    # Volume ratio (taker buy / total volume)
    volume_ratio = np.zeros(n)
    mask = volume > 0
    volume_ratio[mask] = taker_buy_vol[mask] / volume[mask]
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(close_4h, period=21)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        # Align to 1h timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
        st_trend_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE LEVELS
    SIZE_BASE = 0.15    # Base position (low conviction)
    SIZE_HIGH = 0.25    # High conviction
    SIZE_VHIGH = 0.30   # Very high conviction (max)
    
    # ATR stoploss - TIGHTER than current best
    ATR_STOP_MULT = 1.5
    
    # MACD histogram thresholds
    MACD_LONG_MIN = 0.0      # Histogram turning positive
    MACD_SHORT_MAX = 0.0     # Histogram turning negative
    
    # Z-score filter (avoid extremes)
    ZSCORE_MAX = 2.0         # Don't buy if >2 std dev above mean
    ZSCORE_MIN = -2.0        # Don't sell if >2 std dev below mean
    
    # Volume confirmation
    VOLUME_RATIO_LONG = 0.52   # Slight buy pressure
    VOLUME_RATIO_SHORT = 0.48  # Slight sell pressure
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        hist = macd_hist[i]
        hist_prev = macd_hist[i - 1] if i > 0 else 0
        zscore_val = zscore_1h[i]
        st_trend_val = st_trend_1h[i]
        vol_ratio = volume_ratio[i]
        
        # 4h trend filters (MASTER FILTER)
        hma_4h_val = hma_4h_aligned[i]
        st_trend_4h_val = st_trend_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if hma_4h_val > 0 and price > hma_4h_val:
            trend_4h = 1
        elif hma_4h_val > 0 and price < hma_4h_val:
            trend_4h = -1
        
        if st_trend_4h_val == 1:
            trend_4h = max(trend_4h, 1)
        elif st_trend_4h_val == -1:
            trend_4h = min(trend_4h, -1)
        
        # ========== CHECK EXISTING POSITIONS ==========
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
            
            # Stoploss check (1.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE  # Reduce to half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE  # Reduce to half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
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
        
        # ========== ENTRY LOGIC - MACD MOMENTUM + Z-SCORE FILTER ==========
        # LONG: 4h trend up + 1h Supertrend up + MACD histogram positive + Z-score not extreme + volume confirms
        long_condition = (
            trend_4h == 1 and
            st_trend_val == 1 and
            hist > MACD_LONG_MIN and hist_prev <= MACD_LONG_MIN and  # Histogram crossing above 0
            zscore_val < ZSCORE_MAX and  # Not overbought
            vol_ratio >= VOLUME_RATIO_LONG  # Buy pressure
        )
        
        # SHORT: 4h trend down + 1h Supertrend down + MACD histogram negative + Z-score not extreme + volume confirms
        short_condition = (
            trend_4h == -1 and
            st_trend_val == -1 and
            hist < MACD_SHORT_MAX and hist_prev >= MACD_SHORT_MAX and  # Histogram crossing below 0
            zscore_val > ZSCORE_MIN and  # Not oversold
            vol_ratio <= VOLUME_RATIO_SHORT  # Sell pressure
        )
        
        # Determine conviction level
        conviction = 0
        if long_condition or short_condition:
            # High conviction: 4h Supertrend agrees
            if (long_condition and st_trend_4h_val == 1) or (short_condition and st_trend_4h_val == -1):
                conviction = 2
            # Very high conviction: MACD histogram strong + volume strong
            if long_condition and hist > 0.5 * atr and vol_ratio > 0.55:
                conviction = 3
            elif short_condition and hist < -0.5 * atr and vol_ratio < 0.45:
                conviction = 3
            elif conviction < 2:
                conviction = 1
        
        # Assign position size based on conviction
        if long_condition:
            if conviction >= 3:
                size = SIZE_VHIGH
            elif conviction >= 2:
                size = SIZE_HIGH
            else:
                size = SIZE_BASE
            
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            if conviction >= 3:
                size = SIZE_VHIGH
            elif conviction >= 2:
                size = SIZE_HIGH
            else:
                size = SIZE_BASE
            
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals
```

## Last Updated
2026-03-21 18:29
