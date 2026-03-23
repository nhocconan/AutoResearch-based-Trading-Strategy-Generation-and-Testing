# Strategy: mtf_donchian_rsi_volume_atr_v1

## Status
ACTIVE - Sharpe=6.689 | Return=+23403.3% | DD=-4.3%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 6.046 | +1943.5% | -3.1% | 1038 |
| ETHUSDT | 6.681 | +6717.5% | -5.2% | 1033 |
| SOLUSDT | 7.340 | +61549.1% | -4.5% | 939 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 6.815 | +113.0% | -2.7% | 312 |
| ETHUSDT | 7.221 | +250.1% | -4.1% | 326 |
| SOLUSDT | 7.967 | +418.0% | -5.0% | 310 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #017 - Donchian Breakout + RSI Pullback + Volume Confirmation + ATR Stop
====================================================================================
Hypothesis: Donchian channel breakouts (4h) provide cleaner trend signals than MA crossovers.
Combined with RSI pullback entries (1h) and volume confirmation, this should capture
momentum earlier while avoiding false breakouts. ATR trailing stop limits drawdown.

Key differences from previous attempts:
- Donchian(20) breakout instead of KAMA/EMA for pure price action trend
- Volume spike confirmation (>1.5x 20-period avg) to validate breakouts
- RSI(14) pullback entries in trend direction (not counter-trend)
- Z-score(20) filter to avoid entering at >2 std dev extensions
- Discrete position sizing (0.0, ±0.25, ±0.35) to reduce churn costs
- ATR(14) trailing stop at 2.5*ATR with entry price tracking

Why this might beat Sharpe=2.139:
- Donchian captures breakouts earlier than volatility-based channels (Keltner/Bollinger)
- Volume confirmation filters false breakouts (major issue in #008, #010 failures)
- RSI pullback entries reduce chase risk vs breakout entries
- Multi-timeframe logic proven in mtf_hma_rsi_zscore_v1 (Sharpe=5.4)
- Conservative position sizing (max 0.35) controls drawdown
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_rsi_volume_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel - pure price action breakout"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


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


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above moving average"""
    n = len(volume)
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = np.zeros(n)
    
    mask = vol_ma > 0
    spike[mask] = (volume[mask] / vol_ma[mask]) > threshold
    
    return spike


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    volume_spike_1h = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # 4h Donchian for trend filter (resample 1h → 4h)
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
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Donchian
    donchian_upper, donchian_lower = calculate_donchian(h_4h, l_4h, period=20)
    
    # 4h trend direction based on Donchian position
    trend_4h = np.zeros(len(c_4h))
    for i in range(20, len(c_4h)):
        channel_range = donchian_upper[i] - donchian_lower[i]
        if channel_range > 0:
            price_position = (c_4h[i] - donchian_lower[i]) / channel_range
            if price_position > 0.65:
                trend_4h[i] = 1  # Bullish (price in upper 35% of channel)
            elif price_position < 0.35:
                trend_4h[i] = -1  # Bearish (price in lower 35% of channel)
    
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
    SIZE_FULL = 0.35   # Full position with volume confirmation
    SIZE_HALF = 0.25   # Reduced position without volume
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 50   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 50  # Enter short on rally in downtrend
    
    # Z-score thresholds for regime filter
    ZSCORE_MAX = 2.0      # Don't enter if price > 2 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 20, 14, 20)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        vol_spike = volume_spike_1h[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # Z-score filter - avoid entering at extreme deviations
        if abs(zscore_val) > ZSCORE_MAX:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Determine position size based on volume confirmation
        position_size = SIZE_FULL if vol_spike else SIZE_HALF
        
        if trend == 1:  # 4h uptrend
            # RSI pullback entry in uptrend
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry in downtrend
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 07:54
