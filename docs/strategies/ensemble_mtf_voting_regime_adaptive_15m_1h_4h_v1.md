# Strategy: ensemble_mtf_voting_regime_adaptive_15m_1h_4h_v1

## Status
ACTIVE - Sharpe=0.223 | Return=+41.6% | DD=-17.8%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.340 | +6.6% | -15.1% | 218 |
| ETHUSDT | 0.108 | +24.8% | -14.0% | 2 |
| SOLUSDT | 0.903 | +93.4% | -24.2% | 8 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.493 | +2.4% | -5.9% | 164 |
| ETHUSDT | -0.976 | -6.3% | -16.2% | 114 |
| SOLUSDT | -0.778 | -5.4% | -18.4% | 157 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #060 - Triple Timeframe Ensemble with Regime-Adaptive Sizing
==================================================================================================
Hypothesis: Single-indicator strategies whipsaw in choppy markets. 
Combine 3 independent signal types (trend, momentum, mean-reversion) with voting logic.
Only enter when 2/3 signals agree, scale size by agreement strength + volatility regime.

Key innovations:
- 4h Supertrend + HMA for primary trend (stable, low whipsaw)
- 1h MACD histogram for momentum confirmation
- 15m RSI + Z-score for precise entry timing
- Voting: need 2/3 signals aligned for entry
- Regime-adaptive sizing: 0.35 in low vol, 0.20 in high vol
- Stoploss: 2.5*ATR, TP: 2R then trail at 1R
- Discrete signal levels to minimize churn costs

Why this should beat Sharpe=3.653:
- Ensemble reduces false signals (need confirmation)
- Multi-timeframe alignment using mtf_data helper (proven critical)
- Regime detection avoids large positions in high volatility
- Conservative sizing (max 0.35) protects against 2022-style crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_mtf_voting_regime_adaptive_15m_1h_4h_v1"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # EMA calculation
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
    
    # Signal line (EMA of MACD)
    signal_line = np.zeros(n)
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    for i in range(slow + signal, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * (2 / (signal + 1)) + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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


def calculate_bbw_percentile(bbw, lookback=100):
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
    zscore_15m = calculate_zscore(close, period=20)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    _, _, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Get 4h data using mtf_data helper (CRITICAL - proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate 4h indicators
        hma_4h = calculate_hma(close_4h, period=21)
        supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        _, _, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        
        mtf_available = True
    except Exception:
        # Fallback if mtf_data not available
        mtf_available = False
        hma_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.ones(n)
        macd_hist_4h_aligned = np.zeros(n)
        bbw_4h_aligned = np.zeros(n)
    
    # Get 1h data using mtf_data helper
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        # Calculate 1h indicators
        rsi_1h = calculate_rsi(close_1h, period=14)
        _, _, macd_hist_1h = calculate_macd(close_1h, fast=12, slow=26, signal=9)
        
        # Align 1h indicators to 15m timeframe
        rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
        macd_hist_1h_aligned = align_htf_to_ltf(prices, df_1h, macd_hist_1h)
        
        mtf_1h_available = True
    except Exception:
        mtf_1h_available = False
        rsi_1h_aligned = np.zeros(n)
        macd_hist_1h_aligned = np.zeros(n)
    
    # Generate signals with ensemble voting logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_MEDIUM = 0.25
    SIZE_REDUCED = 0.20  # For high volatility regime
    
    # Signal thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    ZSCORE_MAX = 2.0
    MACD_THRESHOLD = 0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # BBW percentile for regime detection
    BBW_HIGH_VOL_PCT = 0.70  # Above this = high volatility regime
    
    first_valid = max(200, 14 * 2, 20, 100)
    
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
        
        # === SIGNAL 1: 4h Trend (Supertrend + HMA) ===
        if mtf_available:
            price_4h = close_4h[min(i // 16, len(close_4h) - 1)] if len(close_4h) > 0 else close[i]
            hma_4h_val = hma_4h_aligned[i]
            st_trend_4h = st_direction_4h_aligned[i]
            macd_4h = macd_hist_4h_aligned[i]
            
            # Determine 4h trend direction
            if hma_4h_val > 0:
                if price_4h > hma_4h_val and st_trend_4h == 1:
                    trend_signal = 1
                elif price_4h < hma_4h_val and st_trend_4h == -1:
                    trend_signal = -1
                else:
                    trend_signal = 0
            else:
                trend_signal = 0
        else:
            trend_signal = 0
        
        # === SIGNAL 2: 1h Momentum (MACD + RSI) ===
        if mtf_1h_available:
            rsi_1h_val = rsi_1h_aligned[i]
            macd_1h = macd_hist_1h_aligned[i]
            
            if macd_1h > MACD_THRESHOLD and rsi_1h_val > 50:
                momentum_signal = 1
            elif macd_1h < MACD_THRESHOLD and rsi_1h_val < 50:
                momentum_signal = -1
            else:
                momentum_signal = 0
        else:
            momentum_signal = 0
        
        # === SIGNAL 3: 15m Entry Timing (RSI + Z-score + MACD) ===
        rsi_15m_val = rsi_15m[i]
        zscore_15m_val = zscore_15m[i]
        macd_15m_val = macd_hist_15m[i]
        
        entry_signal = 0
        if (RSI_LONG_MIN <= rsi_15m_val <= RSI_LONG_MAX and 
            abs(zscore_15m_val) < ZSCORE_MAX and
            macd_15m_val > MACD_THRESHOLD):
            entry_signal = 1
        elif (RSI_SHORT_MIN <= rsi_15m_val <= RSI_SHORT_MAX and 
              abs(zscore_15m_val) < ZSCORE_MAX and
              macd_15m_val < MACD_THRESHOLD):
            entry_signal = -1
        
        # === ENSEMBLE VOTING ===
        # Need at least 2/3 signals aligned for entry
        signal_sum = trend_signal + momentum_signal + entry_signal
        
        # Regime detection (BBW percentile)
        bbw_pct = bbw_pct_15m[i]
        high_vol_regime = bbw_pct > BBW_HIGH_VOL_PCT
        
        # Adjust position size for regime
        if high_vol_regime:
            current_size = SIZE_REDUCED
            current_size_half = SIZE_REDUCED / 2
        else:
            current_size = SIZE_FULL
            current_size_half = SIZE_MEDIUM / 2
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, close[i])
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = current_size_half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -current_size_half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if close[i] > trail_stop:
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
        
        # Entry logic: Need 2/3 signals aligned
        if signal_sum >= 2:  # Bullish consensus
            signals[i] = current_size
            position_side[i] = 1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
            
        elif signal_sum <= -2:  # Bearish consensus
            signals[i] = -current_size
            position_side[i] = -1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 14:18
