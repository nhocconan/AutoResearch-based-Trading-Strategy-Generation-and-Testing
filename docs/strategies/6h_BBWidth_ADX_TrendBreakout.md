# Strategy: 6h_BBWidth_ADX_TrendBreakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.890 | -8.3% | -9.6% | 41 | FAIL |
| ETHUSDT | 0.536 | +37.5% | -6.9% | 35 | PASS |
| SOLUSDT | 0.524 | +45.3% | -8.3% | 21 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.097 | +6.7% | -3.4% | 9 | PASS |
| SOLUSDT | 0.807 | +12.3% | -3.6% | 7 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Bollinger Band width (volatility) and 1d ADX (trend strength) to filter entries
# - Uses Bollinger Band width percentile to identify low volatility regimes (squeeze) on 12h
# - Uses 1d ADX to confirm trend strength (ADX > 25) for breakout direction
# - Enters long when price breaks above upper BB with volume spike in low vol + strong trend
# - Enters short when price breaks below lower BB with volume spike in low vol + strong trend
# - Exits when price crosses back below/above middle BB or volatility expands (BB width > 80th percentile)
# - Designed to capture volatility breakouts after consolidation periods with trend confirmation
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_BBWidth_ADX_TrendBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Bollinger Band width calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 12h Bollinger Bands (20, 2)
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Middle band (SMA20)
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_dev = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma_20 + (2 * std_dev)
    lower_bb = sma_20 - (2 * std_dev)
    # Bollinger Band width
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Calculate 12h BB width percentile rank (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1d ADX (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # Align 12h indicators to 6h timeframe
    bb_width_percentile_6h = align_htf_to_ltf(prices, df_12h, bb_width_percentile)
    middle_bb_6h = align_htf_to_ltf(prices, df_12h, sma_20)
    upper_bb_6h = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_6h = align_htf_to_ltf(prices, df_12h, lower_bb)
    
    # Align 1d ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(bb_width_percentile_6h[i]) or np.isnan(middle_bb_6h[i]) or 
            np.isnan(upper_bb_6h[i]) or np.isnan(lower_bb_6h[i]) or
            np.isnan(adx_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for low volatility regime (BB width < 20th percentile) and strong trend (ADX > 25)
            low_vol_regime = bb_width_percentile_6h[i] < 20
            strong_trend = adx_6h[i] > 25
            
            if low_vol_regime and strong_trend:
                # Long: price breaks above upper BB with volume spike
                if close[i] > upper_bb_6h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower BB with volume spike
                elif close[i] < lower_bb_6h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below middle BB OR volatility expands (BB width > 80th percentile)
            if close[i] < middle_bb_6h[i] or bb_width_percentile_6h[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle BB OR volatility expands (BB width > 80th percentile)
            if close[i] > middle_bb_6h[i] or bb_width_percentile_6h[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 23:01
