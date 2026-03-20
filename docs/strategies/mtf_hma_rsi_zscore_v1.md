# Strategy: mtf_hma_rsi_zscore_v1

## Status
ACTIVE - Sharpe=5.414 | Return=+2871.4% | DD=-7.5%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 4.575 | +428.0% | -6.4% | 3918 |
| ETHUSDT | 5.184 | +884.5% | -4.8% | 3803 |
| SOLUSDT | 6.482 | +7301.6% | -11.2% | 3621 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 4.650 | +49.4% | -2.6% | 1111 |
| ETHUSDT | 6.124 | +111.0% | -4.8% | 1138 |
| SOLUSDT | 7.318 | +172.7% | -3.2% | 1010 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #011 - Multi-Timeframe HMA Trend + RSI/Z-Score Entry
================================================================
Hypothesis: Combining 4h HMA(48) trend filter with 1h RSI(14) + Z-score(20) 
entry signals will capture trends while avoiding extreme volatility entries.
HMA reduces lag vs EMA/SMA, Z-score filters prevent chasing momentum extremes.

Key differences from mtf_supertrend_rsi_v1:
- HMA(48) instead of Supertrend for trend (smoother, less whipsaw)
- Z-score(20) filter to avoid entries during extreme moves
- Dynamic position sizing based on volatility regime
- More conservative sizing (0.20-0.35) to control drawdown

Why this might beat Sharpe=2.501:
- HMA has less lag than EMA, better trend capture
- Z-score filter reduces bad entries during volatility spikes
- Multi-TF approach already proven to work (exp#010)
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_zscore_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=48):
    """
    Calculate Hull Moving Average
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        result = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            result[i] = np.sum(series[i - window + 1:i + 1] * weights)
        return result
    
    close_series = np.array(close)
    wma_half = wma(close_series, half)
    wma_full = wma(close_series, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n) window
    hma = wma(diff, sqrt_n)
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score for volatility regime detection"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    
    # 4h HMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Calculate 4h HMA
    hma_4h = calculate_hma(df_4h['close'].values, period=48)
    
    # Calculate 4h trend direction (price vs HMA)
    trend_4h = np.zeros(len(hma_4h))
    for i in range(len(hma_4h)):
        if hma_4h[i] > 0:
            if df_4h['close'].values[i] > hma_4h[i]:
                trend_4h[i] = 1  # Bullish
            else:
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 40   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 60  # Enter short on rally in downtrend
    RSI_EXIT = 50         # Neutral zone
    
    # Z-score thresholds for volatility filter
    ZSCORE_MAX = 2.0      # Don't enter if price is >2 std from mean
    
    first_valid = max(48, 20, 14)  # Wait for all indicators
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        
        # Volatility filter - don't trade during extreme moves
        if abs(zscore_val) > ZSCORE_MAX:
            signals[i] = 0.0
            continue
        
        if trend == 1:  # 4h uptrend
            if rsi_val < RSI_LONG_ENTRY:
                # Strong pullback - full position
                signals[i] = SIZE_FULL
            elif rsi_val < RSI_EXIT:
                # Moderate pullback - half position
                signals[i] = SIZE_HALF
            else:
                # No pullback - hold or exit
                signals[i] = 0.0
        elif trend == -1:  # 4h downtrend
            if rsi_val > RSI_SHORT_ENTRY:
                # Strong rally - full short
                signals[i] = -SIZE_FULL
            elif rsi_val > RSI_EXIT:
                # Moderate rally - half short
                signals[i] = -SIZE_HALF
            else:
                # No rally - hold or exit
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-21 05:42
