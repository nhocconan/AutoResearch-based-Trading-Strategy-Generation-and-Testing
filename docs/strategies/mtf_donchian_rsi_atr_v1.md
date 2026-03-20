# Strategy: mtf_donchian_rsi_atr_v1

## Status
ACTIVE - Sharpe=5.884 | Return=+3462.4% | DD=-3.9%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 5.061 | +607.1% | -3.6% | 1569 |
| ETHUSDT | 5.735 | +1308.5% | -3.5% | 1470 |
| SOLUSDT | 6.857 | +8471.8% | -4.5% | 1414 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 6.583 | +74.6% | -1.3% | 446 |
| ETHUSDT | 6.904 | +148.4% | -3.5% | 443 |
| SOLUSDT | 5.957 | +145.4% | -3.2% | 395 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #011 - Donchian Breakout + RSI Pullback + ATR Trailing Stop
=======================================================================
Hypothesis: Donchian channels capture trend breakouts with less lag than moving averages.
Combined with RSI pullback entries and ATR-based trailing stops, this should reduce
whipsaw while maintaining trend exposure. BB Width filter avoids extreme volatility.

Key differences from mtf_kama_bb_rsi_v1:
- Donchian(20) breakout instead of KAMA for trend (pure price action)
- ATR trailing stop with 2.5*ATR distance (dynamic risk management)
- BB Width percentile filter to avoid low-volatility traps
- Multi-timeframe: 4h Donchian trend + 1h RSI entries

Why this might beat Sharpe=5.677:
- Donchian breakouts capture momentum earlier than MA crosses
- ATR stops adapt to volatility regime
- BB Width filter avoids choppy consolidation periods
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2
    return upper, lower, middle


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


def calculate_bb_width(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width for volatility regime"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    bb_width = (upper - lower) / mean
    
    return bb_width


def calculate_bb_percentile(close, period=20, std_mult=2.0, lookback=100):
    """Calculate BB Width percentile for regime detection"""
    bb_width = calculate_bb_width(close, period, std_mult)
    n = len(close)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bb_width[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= bb_width[i]) / len(valid)
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_pct_1h = calculate_bb_percentile(close, period=20, std_mult=2.0, lookback=100)
    
    # 4h Donchian for trend filter (resample 1h → 4h)
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
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Donchian
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(h_4h, l_4h, period=20)
    
    # 4h trend direction based on Donchian position
    trend_4h = np.zeros(len(c_4h))
    for i in range(20, len(c_4h)):
        if donchian_upper[i] > 0:
            price_position = (c_4h[i] - donchian_lower[i]) / (donchian_upper[i] - donchian_lower[i])
            if price_position > 0.6:
                trend_4h[i] = 1  # Bullish (price in upper 40% of channel)
            elif price_position < 0.4:
                trend_4h[i] = -1  # Bearish (price in lower 40% of channel)
    
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
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    
    # BB Width percentile thresholds for volatility filter
    BB_PCT_MIN = 0.20     # Don't trade in extremely low vol (consolidation)
    BB_PCT_MAX = 0.85     # Don't trade in extremely high vol (panic/euphoria)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 20, 14, 100)  # Wait for all indicators
    
    # Track recent highs/lows for trailing stop logic
    recent_long_entry_price = np.zeros(n)
    recent_short_entry_price = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_pct_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        bb_pct = bb_pct_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Volatility filter - avoid extreme regimes
        if bb_pct < BB_PCT_MIN or bb_pct > BB_PCT_MAX:
            signals[i] = 0.0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            continue
        
        if trend == 1:  # 4h uptrend
            if rsi_val < RSI_LONG_ENTRY:
                # Pullback entry - full position
                signals[i] = SIZE_FULL
                recent_long_entry_price[i] = price
            elif rsi_val < 50:
                # Moderate pullback - half position
                signals[i] = SIZE_HALF
                recent_long_entry_price[i] = price
            else:
                # Check trailing stop for existing long
                if i > 0 and signals[i - 1] > 0:
                    # Find most recent long entry price
                    entry_idx = max(0, i - 50)
                    entry_prices = recent_long_entry_price[entry_idx:i]
                    valid_entries = entry_prices[entry_prices > 0]
                    if len(valid_entries) > 0:
                        entry_price = valid_entries[-1]
                        stoploss_price = entry_price - ATR_STOP_MULT * atr
                        if price < stoploss_price:
                            signals[i] = 0.0  # Stoploss triggered
                        else:
                            signals[i] = signals[i - 1]  # Hold position
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
        elif trend == -1:  # 4h downtrend
            if rsi_val > RSI_SHORT_ENTRY:
                # Rally entry - full short
                signals[i] = -SIZE_FULL
                recent_short_entry_price[i] = price
            elif rsi_val > 50:
                # Moderate rally - half short
                signals[i] = -SIZE_HALF
                recent_short_entry_price[i] = price
            else:
                # Check trailing stop for existing short
                if i > 0 and signals[i - 1] < 0:
                    # Find most recent short entry price
                    entry_idx = max(0, i - 50)
                    entry_prices = recent_short_entry_price[entry_idx:i]
                    valid_entries = entry_prices[entry_prices > 0]
                    if len(valid_entries) > 0:
                        entry_price = valid_entries[-1]
                        stoploss_price = entry_price + ATR_STOP_MULT * atr
                        if price > stoploss_price:
                            signals[i] = 0.0  # Stoploss triggered
                        else:
                            signals[i] = signals[i - 1]  # Hold position
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-21 05:55
