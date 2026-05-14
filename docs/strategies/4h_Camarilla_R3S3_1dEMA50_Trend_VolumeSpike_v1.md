# Strategy: 4h_Camarilla_R3S3_1dEMA50_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.346 | +33.8% | -8.8% | 235 | PASS |
| ETHUSDT | 0.021 | +21.1% | -9.5% | 217 | PASS |
| SOLUSDT | 0.928 | +105.2% | -14.1% | 176 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.354 | -3.6% | -5.5% | 86 | FAIL |
| ETHUSDT | 0.634 | +13.9% | -6.9% | 78 | PASS |
| SOLUSDT | 0.327 | +9.8% | -8.4% | 62 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 AND price > 1d EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 AND price < 1d EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses the Camarilla pivot point (PP).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 75-150 total trades over 4 years (19-38/year). Works in bull/bear via 1d EMA50 trend filter.

name = "4h_Camarilla_R3S3_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    # We need the previous completed 1d bar's H, L, C
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d_vals) / 3.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 2.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R3, uptrend (price > 1d EMA50), volume confirmation
            if (curr_high > r3_aligned[i] and 
                curr_close > ema_50_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below S3, downtrend (price < 1d EMA50), volume confirmation
            elif (curr_low < s3_aligned[i] and 
                  curr_close < ema_50_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below pivot point (PP)
            if curr_close < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above pivot point (PP)
            if curr_close > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-30 15:26
