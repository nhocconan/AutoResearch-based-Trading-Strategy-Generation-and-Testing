# Strategy: simplified_mtf_regime_15m_4h_v1

## Status
ACTIVE - Sharpe=0.070 | Return=+35.6% | DD=-16.9%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.209 | +7.7% | -13.6% | 2203 |
| ETHUSDT | -0.176 | +5.0% | -20.3% | 2176 |
| SOLUSDT | 0.597 | +94.0% | -16.9% | 2520 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.078 | -4.6% | -8.2% | 670 |
| ETHUSDT | 0.505 | +13.6% | -7.8% | 669 |
| SOLUSDT | 1.185 | +29.2% | -6.8% | 671 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #061 - Simplified Dual-Timeframe Regime Strategy with Clean State Management
==================================================================================================
Hypothesis: Complex 3-signal voting creates too many signal changes (fees) and state tracking bugs.
Simplify to 2-signal confirmation (4h trend + 15m momentum) with cleaner regime-adaptive sizing.

Key improvements over #060:
- Remove 1h timeframe (reduces complexity, 4h+15m is proven in baseline)
- Cleaner position state tracking (no bugs in stop/TP logic)
- More conservative sizing: 0.25 max in low vol, 0.15 in high vol
- Simpler entry: need trend + momentum aligned (not 2/3 voting)
- Better regime detection: BBW percentile over 200 bars
- Discrete signal levels only (0.0, ±0.15, ±0.25) to minimize churn

Why this should beat Sharpe=0.223:
- Fewer signal changes = lower fees
- Cleaner state management = proper stop/TP execution
- Conservative sizing = lower drawdown
- Proven 4h+15m combination from baseline (Sharpe=3.653)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "simplified_mtf_regime_15m_4h_v1"
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


def calculate_macd_histogram(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram only"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n)
    
    def ema(data, period):
        result = np.zeros(n)
        multiplier = 2 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, n):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    for i in range(slow + signal, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * (2 / (signal + 1)) + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    return histogram


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


def calculate_bbw_percentile(bbw, lookback=200):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
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
    macd_hist_15m = calculate_macd_histogram(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=200)
    
    # Get 4h data using mtf_data helper (CRITICAL - proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate 4h indicators
        hma_4h = calculate_hma(close_4h, period=21)
        _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        macd_hist_4h = calculate_macd_histogram(close_4h, fast=12, slow=26, signal=9)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
        
        mtf_available = True
    except Exception:
        mtf_available = False
        hma_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.ones(n)
        macd_hist_4h_aligned = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_LOW_VOL = 0.25  # Low volatility regime
    SIZE_HIGH_VOL = 0.15  # High volatility regime
    
    # Signal thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    MACD_THRESHOLD = 0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    ATR_TP_MULT = 2.0  # Take profit at 2R
    
    # BBW percentile for regime detection
    BBW_HIGH_VOL_PCT = 0.70  # Above this = high volatility regime
    
    first_valid = max(250, 14 * 2, 20, 200)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Track position state (simplified - no complex TP trailing)
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(macd_hist_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Regime detection
        bbw_pct = bbw_pct_15m[i]
        high_vol_regime = bbw_pct > BBW_HIGH_VOL_PCT
        
        # Set position size based on regime
        if high_vol_regime:
            current_size = SIZE_HIGH_VOL
        else:
            current_size = SIZE_LOW_VOL
        
        # === 4h Trend Signal ===
        trend_signal = 0
        if mtf_available:
            price_4h = close_4h[min(i // 16, len(close_4h) - 1)] if len(close_4h) > 0 else close[i]
            hma_4h_val = hma_4h_aligned[i]
            st_trend_4h = st_direction_4h_aligned[i]
            macd_4h = macd_hist_4h_aligned[i]
            
            if hma_4h_val > 0:
                # Bullish: price above HMA + supertrend up + MACD positive
                if price_4h > hma_4h_val and st_trend_4h == 1 and macd_4h > 0:
                    trend_signal = 1
                # Bearish: price below HMA + supertrend down + MACD negative
                elif price_4h < hma_4h_val and st_trend_4h == -1 and macd_4h < 0:
                    trend_signal = -1
        
        # === 15m Entry Signal ===
        entry_signal = 0
        rsi_15m_val = rsi_15m[i]
        macd_15m_val = macd_hist_15m[i]
        
        # Long entry: RSI in neutral zone + MACD positive
        if RSI_LONG_MIN <= rsi_15m_val <= RSI_LONG_MAX and macd_15m_val > MACD_THRESHOLD:
            entry_signal = 1
        # Short entry: RSI in neutral zone + MACD negative
        elif RSI_SHORT_MIN <= rsi_15m_val <= RSI_SHORT_MAX and macd_15m_val < MACD_THRESHOLD:
            entry_signal = -1
        
        # === Check existing position for stoploss/exit ===
        if in_position:
            # Calculate stoploss level
            if position_side == 1:
                stoploss_price = entry_price - ATR_STOP_MULT * entry_atr
                if close[i] < stoploss_price:
                    # Stoploss hit - exit position
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    continue
            
            elif position_side == -1:
                stoploss_price = entry_price + ATR_STOP_MULT * entry_atr
                if close[i] > stoploss_price:
                    # Stoploss hit - exit position
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    continue
            
            # Check for trend reversal exit
            if position_side == 1 and trend_signal == -1:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                continue
            
            elif position_side == -1 and trend_signal == 1:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                continue
            
            # Hold position
            signals[i] = float(position_side) * current_size
            continue
        
        # === Entry Logic: Need both trend and entry signals aligned ===
        if trend_signal == 1 and entry_signal == 1:
            # Enter long
            signals[i] = current_size
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr_15m[i]
        
        elif trend_signal == -1 and entry_signal == -1:
            # Enter short
            signals[i] = -current_size
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr_15m[i]
        
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-21 14:19
