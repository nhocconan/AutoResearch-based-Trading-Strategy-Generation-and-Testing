# Strategy: 6h_12h_1d_camarilla_pivot_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.285 | +10.7% | -14.7% | 316 | DISCARD |
| ETHUSDT | 0.172 | +28.1% | -7.9% | 314 | KEEP |
| SOLUSDT | 0.427 | +51.8% | -21.5% | 272 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.058 | +20.8% | -8.7% | 98 | KEEP |
| SOLUSDT | 0.206 | +8.4% | -11.1% | 100 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot + 12h volume confirmation + 1d trend filter
# - Camarilla levels calculated from prior 12h bar (H12, L12, C12)
# - Long when price closes above R3 with volume > 1.3x 20-bar avg AND 1d close > 1d EMA50
# - Short when price closes below S3 with volume > 1.3x 20-bar avg AND 1d close < 1d EMA50
# - Exit when price returns to Camarilla pivot point (PP)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in ranging/volatile markets (2022-2025)
# - Volume confirmation filters false breakouts
# - 1d trend filter ensures alignment with higher timeframe

name = "6h_12h_1d_camarilla_pivot_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h indicators for Camarilla calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels from prior 12h bar
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    pp_12h = typical_price_12h  # Simplified: PP = (H+L+C)/3
    range_12h = high_12h - low_12h
    r3_12h = pp_12h + (range_12h * 1.1 / 2.0)
    s3_12h = pp_12h - (range_12h * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (completed 12h bar only)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(pp_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price closes above R3 with volume spike and 1d uptrend
            if (prices['close'].iloc[i] > r3_12h_aligned[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price closes below S3 with volume spike and 1d downtrend
            elif (prices['close'].iloc[i] < s3_12h_aligned[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to Camarilla pivot point (PP)
            if position == 1 and prices['close'].iloc[i] < pp_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > pp_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
```

## Last Updated
2026-04-10 03:55
