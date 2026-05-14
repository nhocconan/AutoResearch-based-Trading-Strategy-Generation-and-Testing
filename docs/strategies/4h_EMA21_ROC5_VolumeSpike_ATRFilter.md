# Strategy: 4h_EMA21_ROC5_VolumeSpike_ATRFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.261 | +33.6% | -12.7% | 405 | PASS |
| ETHUSDT | 0.207 | +31.6% | -14.6% | 395 | PASS |
| SOLUSDT | 0.210 | +33.2% | -38.3% | 346 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.718 | -11.7% | -15.6% | 161 | FAIL |
| ETHUSDT | 0.120 | +7.2% | -16.9% | 143 | PASS |
| SOLUSDT | -0.127 | +2.5% | -14.7% | 123 | FAIL |

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
    
    # === Daily ATR for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 4h EMA21 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # === Price Momentum (ROC 5-period) ===
    roc_5 = ((pd.Series(close).pct_change(5) * 100)).values
    
    # === Volume Spike Detection (15-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_spike = volume > (1.8 * vol_ma)  # Strong volume spike
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100  # Need ROC(5), EMA21, ATR14
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(roc_5[i]) or np.isnan(ema_21_4h_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        roc = roc_5[i]
        ema21 = ema_21_4h_aligned[i]
        atr = atr_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when momentum fades or volatility drops ===
        if position == 1:  # Long position
            # Exit when momentum turns negative OR volatility drops significantly
            if roc < 0 or atr < (atr_1d_aligned[i-1] * 0.7 if i > 0 else atr):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when momentum turns positive OR volatility drops significantly
            if roc > 0 or atr < (atr_1d_aligned[i-1] * 0.7 if i > 0 else atr):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Strong positive momentum + price above EMA21 + volume spike
            if roc > 0.5 and price > ema21 and vol_spike:
                signals[i] = 0.30
                position = 1
                continue
            
            # SHORT: Strong negative momentum + price below EMA21 + volume spike
            elif roc < -0.5 and price < ema21 and vol_spike:
                signals[i] = -0.30
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA21_ROC5_VolumeSpike_ATRFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 15:41
