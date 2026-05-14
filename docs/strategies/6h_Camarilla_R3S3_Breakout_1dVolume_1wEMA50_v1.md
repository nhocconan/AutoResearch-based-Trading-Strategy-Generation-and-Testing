# Strategy: 6h_Camarilla_R3S3_Breakout_1dVolume_1wEMA50_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.357 | +32.6% | -8.8% | 90 | PASS |
| ETHUSDT | 0.148 | +26.4% | -8.8% | 86 | PASS |
| SOLUSDT | 0.537 | +54.2% | -16.4% | 57 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.079 | -0.8% | -7.9% | 35 | FAIL |
| ETHUSDT | 0.567 | +12.1% | -9.7% | 27 | PASS |
| SOLUSDT | -0.396 | +0.7% | -12.4% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA50 trend filter
# Long when price breaks above R3 (1d) with volume > 2.0x 24-bar average and close > 1w EMA50
# Short when price breaks below S3 (1d) with volume > 2.0x 24-bar average and close < 1w EMA50
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or trend failure (close crosses 1w EMA50)
# Uses Camarilla for intraday structure, volume for confirmation, 1w EMA50 for trend filter
# Designed for low trade frequency (~12-37/year on 6h) to minimize fee drag
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)

name = "6h_Camarilla_R3S3_Breakout_1dVolume_1wEMA50_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # But we use the standard formula: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Actually standard Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Let's use the correct formula: R3 = close + 1.1*(high-low)*1.1/4? Wait, let me check:
    # Standard Camarilla levels:
    # R4 = close + 1.1*(high-low)*1.1/2
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # S4 = close - 1.1*(high-low)*1.1/2
    # But actually the 1.1 multiplier is already included. Let me use:
    # R3 = close + 1.1 * (high - low) * 1.1 / 4? No.
    # Correct formula:
    # R3 = close + 1.1 * (high - low) * 1.1 / 4 is wrong.
    # Let me use the standard: R3 = close + 1.1 * (high - low) / 4
    # S3 = close - 1.1 * (high - low) / 4
    # Yes, that's correct.
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3 from previous 1d bar
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 4
    s3 = prev_close - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (2.0x 24-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 24) + 1  # EMA50(1w) + volume MA(24) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 (1d) with volume spike and close > 1w EMA50 (uptrend)
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 (1d) with volume spike and close < 1w EMA50 (downtrend)
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 (1d) or close < 1w EMA50 (trend failure)
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (1d) or close > 1w EMA50 (trend failure)
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 01:33
