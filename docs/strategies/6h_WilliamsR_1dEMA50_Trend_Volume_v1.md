# Strategy: 6h_WilliamsR_1dEMA50_Trend_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.189 | +28.4% | -7.9% | 77 | PASS |
| ETHUSDT | 0.131 | +26.1% | -10.5% | 60 | PASS |
| SOLUSDT | 0.973 | +123.7% | -16.9% | 50 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.815 | -1.2% | -7.3% | 24 | FAIL |
| ETHUSDT | 0.658 | +14.9% | -6.6% | 20 | PASS |
| SOLUSDT | -0.941 | -7.6% | -12.5% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extremes work in both bull/bear markets
# 1d EMA50 provides strong trend bias to avoid counter-trend trades
# Volume confirmation > 1.8x 24-period EMA ensures institutional participation
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Williams %R(14) < 80 for short entries, > 20 for long entries (avoiding extreme overextension)

name = "6h_WilliamsR_1dEMA50_Trend_Volume_v1"
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
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.8 * 24-period EMA (moderate filter)
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ema_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 14 periods for Williams %R + 50 for 1d EMA50
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_24[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50: long above EMA50, short below EMA50
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Williams %R crossing above 20 from oversold with volume spike
                if williams_r[i] > -20 and williams_r[i-1] <= -20 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Williams %R crossing below 80 from overbought with volume spike
                if williams_r[i] < -80 and williams_r[i-1] >= -80 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: Williams %R crossing below 80 (overbought) or price below 1d EMA50
            if williams_r[i] < -80 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crossing above 20 (oversold) or price above 1d EMA50
            if williams_r[i] > -20 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 23:18
