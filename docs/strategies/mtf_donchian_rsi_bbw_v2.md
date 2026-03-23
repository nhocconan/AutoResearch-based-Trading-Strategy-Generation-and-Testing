# Strategy: mtf_donchian_rsi_bbw_v2

## Status
ACTIVE - Sharpe=1.869 | Return=+68.3% | DD=-2.5%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 1.292 | +40.1% | -1.6% | 68 |
| ETHUSDT | 2.521 | +76.0% | -1.3% | 82 |
| SOLUSDT | 1.794 | +88.9% | -4.7% | 79 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 1.647 | +10.6% | -0.6% | 25 |
| ETHUSDT | 3.220 | +30.4% | -2.0% | 36 |
| SOLUSDT | 1.060 | +10.2% | -2.2% | 19 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #011 - Donchian Trend + RSI Pullback + BBW Regime Filter
====================================================================================
Hypothesis: Donchian channels (20-period high/low) capture crypto breakouts better than 
EMA/HMA because they directly measure price extremes. Crypto trends often continue after
breaking 20-period highs/lows. Combine with RSI pullback entries (proven in #005, #007)
and Bollinger Band Width percentile for regime detection.

Key differences from current best (#005 EMA+RSI+Z-score):
- Donchian(20) trend instead of EMA - captures breakouts more directly
- BBW percentile filter instead of Z-score - detects volatility regime changes
- RSI pullback entries (same as #005 but with Donchian trend)
- 4h Donchian trend + 1h RSI entries (proven MTF structure)
- Trailing stoploss at 2*ATR, take profit at 2R (reduce to half)
- Discrete signal levels: 0.0, ±0.25, ±0.35 to minimize churn costs

Why this might beat Sharpe=5.525:
- Donchian channels work exceptionally well for crypto's momentum-driven trends
- BBW percentile avoids trading during extreme volatility (compression = breakout soon,
  expansion = trend exhaustion)
- RSI pullback entries proven effective in #005 and #007
- Multi-timeframe structure proven to 2x Sharpe vs single timeframe
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_rsi_bbw_v2"
timeframe = "1h"
leverage = 1.0


def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel (20-period high/low)
    Upper = highest high of last 20 periods
    Lower = lowest low of last 20 periods
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = max(high[i - period + 1:i + 1])
        lower[i] = min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
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
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    rsi = np.zeros(n)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n - 1)
    loss = np.zeros(n - 1)
    
    gain[delta > 0] = delta[delta > 0]
    loss[delta < 0] = -delta[delta < 0]
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period - 1] = np.mean(gain[:period])
    avg_loss[period - 1] = np.mean(loss[:period])
    
    for i in range(period, n):
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
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    
    for i in range(period - 1, n):
        middle[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i] * 100
        else:
            bandwidth[i] = 0
    
    return middle, upper, lower, bandwidth


def calculate_bbw_percentile(bandwidth, lookback=100):
    """
    Calculate BBW percentile over lookback period
    Low percentile (< 20) = squeeze (breakout coming)
    High percentile (> 80) = expansion (trend exhaustion)
    """
    n = len(bandwidth)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        if np.all(bandwidth[i - lookback + 1:i + 1] == 0):
            percentile[i] = 50
        else:
            rank = np.sum(bandwidth[i - lookback + 1:i + 1] <= bandwidth[i])
            percentile[i] = rank / lookback * 100
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_mid_1h, bb_up_1h, bb_low_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
    
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
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    n_4h = len(c_4h)
    
    # Calculate 4h Donchian for trend
    donch_up_4h, donch_low_4h, donch_mid_4h = calculate_donchian(h_4h, l_4h, period=20)
    
    # 4h trend direction based on Donchian breakout
    trend_4h = np.zeros(n_4h)
    for i in range(20, n_4h):
        if c_4h[i] > donch_mid_4h[i] and c_4h[i] > c_4h[i - 1]:
            trend_4h[i] = 1  # Bullish (above middle, making higher highs)
        elif c_4h[i] < donch_mid_4h[i] and c_4h[i] < c_4h[i - 1]:
            trend_4h[i] = -1  # Bearish (below middle, making lower lows)
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Map 4h Donchian levels back to 1h for reference
    donch_mid_1h = np.zeros(n)
    donch_up_1h_map = np.zeros(n)
    donch_low_1h_map = np.zeros(n)
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(donch_mid_4h):
            donch_mid_1h[i] = donch_mid_4h[idx_4h]
            donch_up_1h_map[i] = donch_up_4h[idx_4h]
            donch_low_1h_map[i] = donch_low_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 40   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 60  # Enter short on pullback in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # BBW percentile regime filter
    BBW_MIN_PCT = 15  # Avoid extreme squeeze (< 15th percentile)
    BBW_MAX_PCT = 85  # Avoid extreme expansion (> 85th percentile)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(100, 20, 14, 28)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    initial_stop = np.zeros(n)  # Track initial stoploss level
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bbw_pct_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        bbw_pct = bbw_pct_1h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # BBW regime filter - avoid extreme volatility regimes
        if bbw_pct < BBW_MIN_PCT or bbw_pct > BBW_MAX_PCT:
            if i > 0 and position_side[i - 1] != 0:
                # Hold existing position but don't add
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else price
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
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
                    # Reduce to half position
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
                
                # RSI exit signal (overbought)
                if rsi_val > RSI_EXIT_LONG:
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
                    # Reduce to half position
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    initial_stop[i] = prev_stop
                    continue
                
                # RSI exit signal (oversold)
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_stop[i] = 0
                    continue
        
        # Entry logic with RSI pullback confirmation
        position_size = SIZE_FULL
        
        if trend == 1:  # 4h uptrend + BBW regime OK
            # RSI pullback entry in uptrend (RSI dips to 40-50 then rises)
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                # Check previous bar for RSI turning up
                if i > 0 and rsi_1h[i-1] < rsi_val:
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    initial_stop[i] = price - ATR_STOP_MULT * atr
                else:
                    # Hold existing position
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
            else:
                # Hold or exit
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
                    
        elif trend == -1:  # 4h downtrend + BBW regime OK
            # RSI pullback entry in downtrend (RSI rises to 50-60 then falls)
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                # Check previous bar for RSI turning down
                if i > 0 and rsi_1h[i-1] > rsi_val:
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    initial_stop[i] = price + ATR_STOP_MULT * atr
                else:
                    # Hold existing position
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
            else:
                # Hold or exit
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
2026-03-21 08:49
