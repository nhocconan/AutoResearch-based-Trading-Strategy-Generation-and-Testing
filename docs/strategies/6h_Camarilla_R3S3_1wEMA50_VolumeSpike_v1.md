# Strategy: 6h_Camarilla_R3S3_1wEMA50_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.421 | +14.3% | -3.6% | 131 | DISCARD |
| ETHUSDT | 0.309 | +28.9% | -3.5% | 101 | KEEP |
| SOLUSDT | 0.105 | +24.7% | -5.8% | 73 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.328 | +16.5% | -2.4% | 38 | KEEP |
| SOLUSDT | -0.172 | +4.8% | -5.6% | 32 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with weekly trend filter and volume spike confirmation
# Long when price breaks above R3 AND weekly close > weekly EMA50 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below S3 AND weekly close < weekly EMA50 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price retraces to the Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly EMA50 provides strong trend filter between 6h and 1d for better regime adaptation
# Volume spike threshold increased to 2.0x to reduce false breakouts and lower trade frequency
# Pivot exit works in ranging markets and captures mean reversion after breakout failure

name = "6h_Camarilla_R3S3_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for 6h timeframe (based on previous bar)
    # Camarilla: Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1 / 2
    # S3 = Pivot - (H - L) * 1.1 / 2
    pivot = (high + low + close) / 3.0
    r3 = pivot + (high - low) * 1.1 / 2.0
    s3 = pivot - (high - low) * 1.1 / 2.0
    
    # Shift by 1 to use only completed bar data (no look-ahead)
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    pivot_prev = np.roll(pivot, 1)
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan
    pivot_prev[0] = np.nan
    
    # Get weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_prev[i]) or np.isnan(s3_prev[i]) or 
            np.isnan(pivot_prev[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: Break above R3 AND uptrend AND volume spike
            if close[i] > r3_prev[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 AND downtrend AND volume spike
            elif close[i] < s3_prev[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to pivot point (mean reversion)
            if close[i] <= pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to pivot point (mean reversion)
            if close[i] >= pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 20:22
