# Strategy: 4h_DonchianBreakout_VolumeSpike_LowVol

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.032 | +19.2% | -8.9% | 91 | FAIL |
| ETHUSDT | 0.057 | +22.3% | -10.8% | 92 | PASS |
| SOLUSDT | 0.840 | +109.4% | -21.2% | 94 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.198 | +8.8% | -19.2% | 5 | PASS |
| SOLUSDT | 0.110 | +6.5% | -24.1% | 2 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4x ATR for volatility (using 4h data)
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === Daily data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR for volatility regime
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period Donchian channels on daily
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # === Daily volume spike detection ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # === 4h EMA(20) for exit signal ===
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(ema_20_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_4h[i]  # Use 4h close for entry/exit logic
        upper_level = upper_20_aligned[i]
        lower_level = lower_20_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        ema_20_val = ema_20_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 4h EMA(20)
            if price < ema_20_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 4h EMA(20)
            if price > ema_20_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above daily upper Donchian with 4h volume spike and low volatility regime
            if (price > upper_level and 
                volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and  # 4h volume spike
                atr_4h_val < atr_1d_val * 1.2):  # Low volatility regime (4h ATR < 1.2x daily ATR)
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below daily lower Donchian with 4h volume spike and low volatility regime
            elif (price < lower_level and 
                  volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and  # 4h volume spike
                  atr_4h_val < atr_1d_val * 1.2):  # Low volatility regime (4h ATR < 1.2x daily ATR)
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_VolumeSpike_LowVol"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 16:21
