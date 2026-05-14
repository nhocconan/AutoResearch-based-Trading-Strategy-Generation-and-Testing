# Strategy: 4h_1wBBWidth_1dSupertrend_Breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.137 | +2.8% | -3.3% | 37 | FAIL |
| ETHUSDT | 0.445 | +33.0% | -5.5% | 84 | PASS |
| SOLUSDT | 0.158 | +25.2% | -4.5% | 36 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.348 | +8.2% | -4.6% | 47 | PASS |
| SOLUSDT | 0.000 | +0.0% | 0.0% | 0 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Supertrend for trend direction and 1w Bollinger Band width for volatility regime
# - Uses 1w Bollinger Band width percentile to identify low volatility regimes (squeeze) - contrarian indicator
# - Uses 1d Supertrend to confirm trend direction and strength
# - Enters long when price breaks above 1d high with volume spike in low vol + bullish Supertrend
# - Enters short when price breaks below 1d low with volume spike in low vol + bearish Supertrend
# - Exits when price crosses back below/above 1d close or volatility expands (BB width > 80th percentile)
# - Designed to capture volatility breakouts after weekly consolidation with daily trend confirmation
# - Target: 80-160 total trades over 4 years (20-40/year) with 0.25 position sizing

name = "4h_1wBBWidth_1dSupertrend_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for 1d high/low and Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 1w data for Bollinger Band width calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d high and low for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Supertrend (10, 3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, atr_period)
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + multiplier * atr
    basic_lb = (high_1d + low_1d) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(basic_ub)):
        if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close_1d)
    supertrend[0] = final_ub[0]
    for i in range(1, len(supertrend)):
        if supertrend[i-1] == final_ub[i-1]:
            if close_1d[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
        else:
            if close_1d[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    # Supertrend direction: 1 for uptrend, -1 for downtrend
    supertrend_dir = np.where(close_1d > supertrend, 1, -1)
    
    # Calculate 1w Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Middle band (SMA20)
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_dev = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma_20 + (2 * std_dev)
    lower_bb = sma_20 - (2 * std_dev)
    # Bollinger Band width
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Calculate 1w BB width percentile rank (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1d indicators to 4h timeframe
    high_1d_4h = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_4h = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_4h = align_htf_to_ltf(prices, df_1d, close_1d)
    supertrend_dir_4h = align_htf_to_ltf(prices, df_1d, supertrend_dir)
    
    # Align 1w BB width percentile to 4h timeframe
    bb_width_percentile_4h = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(high_1d_4h[i]) or np.isnan(low_1d_4h[i]) or np.isnan(close_1d_4h[i]) or
            np.isnan(supertrend_dir_4h[i]) or np.isnan(bb_width_percentile_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for low volatility regime (BB width < 20th percentile) and bullish/bearish Supertrend
            low_vol_regime = bb_width_percentile_4h[i] < 20
            bullish_trend = supertrend_dir_4h[i] == 1
            bearish_trend = supertrend_dir_4h[i] == -1
            
            if low_vol_regime:
                # Long: price breaks above 1d high with volume spike in bullish trend
                if bullish_trend and close[i] > high_1d_4h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1d low with volume spike in bearish trend
                elif bearish_trend and close[i] < low_1d_4h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below 1d close OR volatility expands (BB width > 80th percentile)
            if close[i] < close_1d_4h[i] or bb_width_percentile_4h[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d close OR volatility expands (BB width > 80th percentile)
            if close[i] > close_1d_4h[i] or bb_width_percentile_4h[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 23:17
