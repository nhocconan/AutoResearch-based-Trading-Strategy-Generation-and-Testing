# Strategy: 4h_Donchian20_Breakout_HMA21_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.181 | +12.9% | -15.7% | 192 | FAIL |
| ETHUSDT | 0.280 | +34.5% | -11.4% | 174 | PASS |
| SOLUSDT | 0.450 | +56.6% | -21.1% | 166 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.482 | +12.5% | -11.9% | 74 | PASS |
| SOLUSDT | 0.709 | +16.6% | -8.8% | 58 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + HMA(21) Trend + Volume Spike
Hypothesis: Donchian breakouts capture strong momentum moves. HMA(21) filters for trend alignment,
volume spike confirms institutional participation. Works in bull (breakouts up) and bear (breakouts down)
by taking both long and short signals. Designed for 4h timeframe with tight entry conditions
to achieve 20-50 trades/year per symbol, minimizing fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(series).ewm(span=half, adjust=False, min_periods=half).mean().values
    wma1 = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values
    raw_hma = 2 * wma2 - wma1
    hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False, min_periods=sqrt).mean().values
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h high/low
    donchian_high = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 15m timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate HMA(21) on 4h close for trend
    hma_21_4h = calculate_hma(df_4h['close'].values, 21)
    hma_21_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_21_4h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20), HMA, volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(hma_21_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        hma_trend = hma_21_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND volume spike AND price > HMA (uptrend)
            long_entry = (curr_close > donchian_high_aligned[i]) and vol_spike and (curr_close > hma_trend)
            # Short: price breaks below Donchian low AND volume spike AND price < HMA (downtrend)
            short_entry = (curr_close < donchian_low_aligned[i]) and vol_spike and (curr_close < hma_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Donchian low OR price crosses below HMA (trend change)
            if (curr_close < donchian_low_aligned[i]) or (curr_close < hma_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian high OR price crosses above HMA (trend change)
            if (curr_close > donchian_high_aligned[i]) or (curr_close > hma_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_HMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 05:35
