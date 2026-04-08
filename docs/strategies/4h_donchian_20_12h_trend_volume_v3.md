# Strategy: 4h_donchian_20_12h_trend_volume_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.344 | +39.4% | -17.2% | 82 | PASS |
| ETHUSDT | -0.017 | +16.2% | -18.7% | 89 | FAIL |
| SOLUSDT | 0.894 | +156.6% | -30.1% | 83 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.388 | -8.5% | -12.3% | 35 | FAIL |
| SOLUSDT | 0.429 | +13.5% | -10.5% | 28 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_donchian_20_12h_trend_volume_v3
Hypothesis: On 4-hour timeframe, use Donchian(20) breakouts with trend filter from 12-hour EMA200 and volume confirmation. Enter long on upper band breakout in uptrend with volume > 1.5x average, short on lower band breakdown in downtrend with volume > 1.5x average. Exit on opposite band touch. Designed for low frequency (19-50 trades/year) to avoid fee drift while capturing trend continuation. Uses 12h trend filter for better alignment with 4h timeframe compared to daily. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) by using 12h trend filter.
Improvements: Uses 12h EMA200 for trend filter (better aligned with 4h), volume confirmation filter, and ATR volatility filter to avoid chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_12h_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    d_close = df_12h['close'].values
    d_ema200 = pd.Series(d_close).ewm(span=200, adjust=False).mean().values
    d_ema200_aligned = align_htf_to_ltf(prices, df_12h, d_ema200)
    
    # Calculate 40-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    # ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio (current ATR / 50-period average ATR) to detect low volatility
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma  # High ratio = high volatility, good for breakouts
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if 12h EMA200 not available
        if np.isnan(d_ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs 12h EMA200
        uptrend = close[i] > d_ema200_aligned[i]
        downtrend = close[i] < d_ema200_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 40-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        # Volatility filter: require ATR ratio > 0.8 (avoid extremely low volatility periods)
        vol_filter = atr_ratio[i] > 0.8 if not np.isnan(atr_ratio[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below lower Donchian(20)
            # Calculate Donchian lower band for last 20 periods
            if i >= 20:
                donchian_low = np.min(low[i-20:i])
                if close[i] <= donchian_low:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above upper Donchian(20)
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                if close[i] >= donchian_high:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need at least 20 periods for Donchian calculation
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                donchian_low = np.min(low[i-20:i])
                
                # Long entry: price breaks above upper Donchian(20) in uptrend with volume confirmation and volatility filter
                long_entry = (close[i] > donchian_high) and uptrend and vol_confirm and vol_filter
                # Short entry: price breaks below lower Donchian(20) in downtrend with volume confirmation and volatility filter
                short_entry = (close[i] < donchian_low) and downtrend and vol_confirm and vol_filter
                
                if long_entry:
                    position = 1
                    signals[i] = 0.25
                elif short_entry:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 17:39
