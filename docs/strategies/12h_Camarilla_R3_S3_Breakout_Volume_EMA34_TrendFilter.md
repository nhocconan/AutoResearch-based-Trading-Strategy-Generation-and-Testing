# Strategy: 12h_Camarilla_R3_S3_Breakout_Volume_EMA34_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.373 | +36.9% | -7.4% | 82 | PASS |
| ETHUSDT | 0.101 | +24.6% | -13.2% | 78 | PASS |
| SOLUSDT | 0.577 | +68.1% | -20.3% | 81 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.797 | -1.5% | -6.2% | 33 | FAIL |
| ETHUSDT | 0.320 | +10.5% | -5.8% | 29 | PASS |
| SOLUSDT | -0.538 | -2.8% | -16.6% | 27 | FAIL |

## Code
```python
# 12h_Camarilla_R3_S3_Breakout_Volume_EMA34_TrendFilter
# Hypothesis: 12h timeframe with Camarilla breakouts, EMA34 trend filter, and volume confirmation
# targets 25-35 trades/year. Uses 1d HTF for levels/trend, reducing false breakouts.
# Works in bull/bear: EMA34 filters trend direction, volume confirms strength.
# Conservative position sizing (0.25) limits drawdown.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (HTF for key levels and trend) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate 1d Camarilla pivot levels (using previous day's OHLC) ===
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    camarilla_base = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_range = prev_high_1d - prev_low_1d
    
    # Resistance and Support levels
    r3 = camarilla_base + camarilla_range * 1.1 / 4
    s3 = camarilla_base - camarilla_range * 1.1 / 4
    
    # Align to 12h timeframe
    camarilla_base_aligned = align_htf_to_ltf(prices, df_1d, camarilla_base)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation (12h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S3 (stop) or hits R3*1.3 (take profit)
            if price < s3_val or price > r3_val * 1.3:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R3 (stop) or hits S3*0.7 (take profit)
            if price > r3_val or price < s3_val * 0.7:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R3 with volume AND above 1d EMA34 (uptrend)
            if (price > r3_val) and (price > ema_34_1d_val) and (vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S3 with volume AND below 1d EMA34 (downtrend)
            elif (price < s3_val) and (price < ema_34_1d_val) and (vol_ratio_val > 1.8):
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

name = "12h_Camarilla_R3_S3_Breakout_Volume_EMA34_TrendFilter"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-16 18:00
