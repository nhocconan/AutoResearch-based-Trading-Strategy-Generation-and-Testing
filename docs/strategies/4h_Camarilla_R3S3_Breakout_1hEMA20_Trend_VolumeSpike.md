# Strategy: 4h_Camarilla_R3S3_Breakout_1hEMA20_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.008 | +20.7% | -10.2% | 321 | PASS |
| ETHUSDT | 0.366 | +37.8% | -8.2% | 280 | PASS |
| SOLUSDT | 0.132 | +26.3% | -14.0% | 228 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.534 | -5.9% | -9.1% | 123 | FAIL |
| ETHUSDT | 1.342 | +25.0% | -5.2% | 104 | PASS |
| SOLUSDT | 0.372 | +10.7% | -7.2% | 81 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1h EMA20 trend filter and volume spike
# Camarilla pivot levels provide precise intraday support/resistance
# 1h EMA20 ensures alignment with short-term trend to avoid counter-trend trades
# Volume spike confirmation filters false breakouts
# Works in bull markets (breakout above R3 + 1h EMA20 up) and bear markets (breakout below S3 + 1h EMA20 down)
# Uses discrete position sizing (0.25) to balance return and drawdown control

name = "4h_Camarilla_R3S3_Breakout_1hEMA20_Trend_VolumeSpike"
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
    
    # 1h data for trend filter (EMA20)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # 1h EMA20 calculation
    ema_20_1h = pd.Series(df_1h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)
    
    # 1d data for Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1h EMA20
        uptrend = close[i] > ema_20_1h_aligned[i]
        downtrend = close[i] < ema_20_1h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Camarilla R3 with volume confirmation and uptrend
            if high[i] > camarilla_r3_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Camarilla S3 with volume confirmation and downtrend
            elif low[i] < camarilla_s3_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 (reversal) OR trend changes to downtrend
            if low[i] < camarilla_s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 (reversal) OR trend changes to uptrend
            if high[i] > camarilla_r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 03:35
