# Strategy: mtf_donchian_rsi_bbw_v1

## Status
ACTIVE - Sharpe=4.711 | Return=+4834.8% | DD=-6.0%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 4.028 | +523.0% | -4.2% | 491 |
| ETHUSDT | 4.214 | +1097.9% | -7.2% | 556 |
| SOLUSDT | 5.890 | +12883.4% | -6.6% | 688 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 2.889 | +39.2% | -3.5% | 123 |
| ETHUSDT | 4.403 | +101.8% | -3.2% | 187 |
| SOLUSDT | 5.805 | +203.4% | -4.7% | 223 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #007 - Donchian Trend + RSI Pullback + Bollinger Width Filter
====================================================================================
Hypothesis: Replace EMA trend with Donchian channel breakout (price above 20-period high
indicates strong momentum trend). Use RSI pullback entries within the trend direction
(proven in #005). Add Bollinger Band width filter to only trade when volatility is
expanding (avoid low-volatility chop). Keep Z-score filter for extreme valuations.

Why this might beat Sharpe=5.525:
- Donchian breakout captures strong momentum trends better than EMA (less lag)
- RSI pullback entries work well for mean reversion within trend (#005 proven)
- Bollinger width filter avoids trading during low-volatility consolidation
- ATR-based stoploss protects against adverse moves
- Multi-timeframe: 4h trend + 1h entries (proven architecture)
- Discrete signal levels (0.0, ±0.20, ±0.35) minimize churn costs
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_rsi_bbw_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian_trend(high, low, close, period=20):
    """
    Calculate Donchian Channel trend signal
    Returns: 1 if price > upper band (bullish), -1 if price < lower band (bearish), 0 otherwise
    """
    n = len(close)
    trend = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        
        mid = (upper[i] + lower[i]) / 2
        
        if close[i] > upper[i]:
            trend[i] = 1
        elif close[i] < lower[i]:
            trend[i] = -1
        elif close[i] > mid:
            trend[i] = 1
        elif close[i] < mid:
            trend[i] = -1
    
    return trend, upper, lower


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    rsi = np.zeros(n)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth"""
    n = len(close)
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
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
    
    return upper, lower, middle, bandwidth


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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        mean = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    bb_upper_1h, bb_lower_1h, bb_mid_1h, bb_bw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # 4h indicators for trend (resample 1h → 4h)
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
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    n_4h = len(c_4h)
    
    # Calculate 4h Donchian trend
    trend_4h, donchian_upper_4h, donchian_lower_4h = calculate_donchian_trend(h_4h, l_4h, c_4h, period=20)
    
    # Calculate 4h Bollinger bandwidth for volatility regime
    _, _, _, bb_bw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    bb_bw_1h_from_4h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
            bb_bw_1h_from_4h[i] = bb_bw_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45    # Buy pullback in uptrend when RSI drops to 45
    RSI_SHORT_ENTRY = 55   # Sell pullback in downtrend when RSI rises to 55
    RSI_EXIT_LONG = 65     # Exit long when RSI reaches 65 (overbought)
    RSI_EXIT_SHORT = 35    # Exit short when RSI reaches 35 (oversold)
    
    # Z-score filter thresholds
    ZSCORE_MAX = 2.0    # Don't enter if price is > 2 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # Bollinger bandwidth filter (percentile-based)
    BB_BW_MIN = 0.02    # Minimum bandwidth to trade (avoid low vol)
    BB_BW_PERCENTILE = 40  # Only trade when BB width is above 40th percentile
    
    # Calculate BB bandwidth percentile over lookback
    bb_bw_percentile = np.zeros(n)
    lookback = 100
    for i in range(lookback - 1, n):
        bb_bw_percentile[i] = np.percentile(bb_bw_1h[i - lookback + 1:i + 1], 40)
    
    first_valid = max(60, 20, 14, 28, lookback)
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(bb_bw_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        bb_bw = bb_bw_1h[i]
        bb_bw_4h_val = bb_bw_1h_from_4h[i]
        bb_bw_thresh = bb_bw_percentile[i]
        
        # Z-score filter - don't enter at extreme valuations
        if abs(zscore_val) > ZSCORE_MAX:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Bollinger bandwidth filter - avoid low volatility periods
        if bb_bw < BB_BW_MIN or bb_bw_4h_val < BB_BW_MIN:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                
                # Stoploss check (2*ATR against position)
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
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
                    continue
                
                # RSI exit signal (overbought)
                if rsi_val >= RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                
                # Stoploss check (2*ATR against position)
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
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
                    continue
                
                # RSI exit signal (oversold)
                if rsi_val <= RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # Generate new entries based on trend + RSI pullback
        if trend == 1:  # 4h uptrend (Donchian breakout)
            # RSI pullback entry (RSI drops to 45 in uptrend)
            if rsi_val <= RSI_LONG_ENTRY:
                if position_side[i - 1] != -1:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            else:
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend (Donchian breakdown)
            # RSI pullback entry (RSI rises to 55 in downtrend)
            if rsi_val >= RSI_SHORT_ENTRY:
                if position_side[i - 1] != 1:
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            else:
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 08:43
