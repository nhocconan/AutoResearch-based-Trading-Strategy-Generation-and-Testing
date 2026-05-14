# Strategy: 6h_12h_camarilla_breakout_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.047 | +21.7% | -11.7% | 114 | PASS |
| ETHUSDT | 0.205 | +31.4% | -15.5% | 113 | PASS |
| SOLUSDT | 0.770 | +114.8% | -33.0% | 104 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.228 | +2.9% | -6.8% | 45 | FAIL |
| ETHUSDT | 0.534 | +15.3% | -9.1% | 39 | PASS |
| SOLUSDT | 0.190 | +8.6% | -13.6% | 38 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Uses 12h HTF for trend direction (EMA50 > EMA200 = uptrend, < = downtrend)
# - 6h Camarilla pivot levels (R3, S3, R4, S4) from prior 12h bar
# - Long on break above R4 in uptrend, short on break below S4 in downtrend
# - Volume confirmation: current 6h volume > 1.3x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)

name = "6h_12h_camarilla_breakout_trend_v1"
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
    open_time = prices['open_time'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMAs for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 12h Camarilla pivot levels from prior bar
    # Typical price = (H + L + C) / 3
    typical_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3_12h = typical_12h + range_12h * 1.1 / 4
    s3_12h = typical_12h - range_12h * 1.1 / 4
    r4_12h = typical_12h + range_12h * 1.1 / 2
    s4_12h = typical_12h - range_12h * 1.1 / 2
    
    # Align all 12h data to 6h timeframe (wait for completed 12h bar)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s4_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Trend filter: 12h EMA50 > EMA200 = uptrend, < = downtrend
        uptrend = ema_50_12h_aligned[i] > ema_200_12h_aligned[i]
        downtrend = ema_50_12h_aligned[i] < ema_200_12h_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price closes below 12h EMA50 (trend change)
            if close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 12h EMA50 (trend change)
            if close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above R4 in uptrend
                if uptrend and close[i] > r4_12h_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below S4 in downtrend
                elif downtrend and close[i] < s4_12h_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals
```

## Last Updated
2026-04-09 18:42
