# Strategy: 4h_ATRTrailingStop_1dCamarilla_R3S3_Breakout_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.466 | -2.9% | -17.0% | 225 | DISCARD |
| ETHUSDT | 0.476 | +51.9% | -15.4% | 199 | KEEP |
| SOLUSDT | 0.686 | +96.9% | -22.1% | 166 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.150 | +7.7% | -10.9% | 72 | KEEP |
| SOLUSDT | 0.074 | +6.3% | -11.7% | 61 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ATR Trailing Stop + 1d Camarilla Pivot Breakout + Volume Spike
# Uses ATR-based trailing stop to let winners run while cutting losses quickly
# Camarilla R3/S3 breakouts with volume confirmation capture momentum moves
# Works in bull markets by buying breakouts above R3 and in bear markets by selling breakdowns below S3
# ATR stoploss adapts to volatility, reducing whipsaw in ranging markets
# Targets 20-50 trades/year (80-200 total over 4 years) for 4h timeframe

name = "4h_ATRTrailingStop_1dCamarilla_R3S3_Breakout_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * camarilla_range / 2
    s3_1d = close_1d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate ATR(14) for trailing stop and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0  # For long trailing stop
    lowest_low_since_entry = 0.0    # For short trailing stop
    
    # Start after warmup (need enough for ATR and volume MA)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(atr[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 + volume spike
            if close[i] > r3_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: Price breaks below S3 + volume spike
            elif close[i] < s3_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_high_since_entry:
                highest_high_since_entry = high[i]
            
            # ATR trailing stop: exit if price drops 2.5*ATR from highest high
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            if close[i] < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_low_since_entry:
                lowest_low_since_entry = low[i]
            
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest low
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            if close[i] > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 15:21
