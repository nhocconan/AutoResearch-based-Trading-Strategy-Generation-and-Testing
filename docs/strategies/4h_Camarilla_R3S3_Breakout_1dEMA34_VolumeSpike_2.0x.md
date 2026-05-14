# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_2.0x

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.536 | +37.6% | -6.8% | 266 | PASS |
| ETHUSDT | 0.092 | +24.1% | -6.5% | 246 | PASS |
| SOLUSDT | 0.278 | +35.4% | -8.9% | 221 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.271 | -1.8% | -4.4% | 102 | FAIL |
| ETHUSDT | 1.580 | +24.6% | -4.6% | 89 | PASS |
| SOLUSDT | 1.192 | +17.9% | -3.2% | 69 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R3/S3) breakout with 1d EMA34 trend filter and volume spike (2.0x)
# Long when price breaks above 4h Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period average
# Short when price breaks below 4h Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period average
# Exit when price crosses 4h Camarilla midpoint (P) OR 1d EMA34 filter reverses
# Uses Camarilla pivot levels from 4h OHLC for responsiveness + volume confirmation to reduce false breakouts
# Designed for 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag
# Timeframe: 4h (primary)

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_2.0x"
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
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels (based on previous bar's OHLC)
    # Camarilla: P = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We use previous bar's values to avoid look-ahead
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    # Set first bar to NaN (no previous bar)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_p = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    camarilla_r3 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 2.0
    camarilla_s3 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 2.0
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_4h, camarilla_p)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation on 4h (threshold: 2.0x for balanced frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot P OR price < EMA34 (trend weakening)
            if close[i] < camarilla_p_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot P OR price > EMA34 (trend weakening)
            if close[i] > camarilla_p_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 14:44
