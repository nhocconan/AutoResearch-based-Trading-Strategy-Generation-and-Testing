# Strategy: 6h_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.405 | +32.1% | -4.2% | 116 | PASS |
| ETHUSDT | 0.046 | +22.6% | -8.7% | 100 | PASS |
| SOLUSDT | 0.833 | +72.0% | -8.5% | 73 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.599 | -6.2% | -6.7% | 38 | FAIL |
| ETHUSDT | 0.979 | +14.1% | -5.7% | 30 | PASS |
| SOLUSDT | -0.704 | -0.2% | -12.2% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with weekly trend filter and volume confirmation
# Camarilla pivots from 1d: R3/S3 are strong reversal levels, R4/S4 are breakout levels
# Long when price breaks above R4 with 1d close > weekly EMA(34) and volume spike
# Short when price breaks below S4 with 1d close < weekly EMA(34) and volume spike
# Uses discrete position sizing (0.25) to minimize fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe

name = "6h_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (based on previous day's OHLC)
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate weekly EMA(34) for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla calculation and volume MA)
    start_idx = 50  # buffer for 20-period volume MA and 1d shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R4 + 1d close > weekly EMA + volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + 1d close < weekly EMA + volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below R3 (reversal signal) or weekly trend breaks
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above S3 (reversal signal) or weekly trend breaks
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 13:24
