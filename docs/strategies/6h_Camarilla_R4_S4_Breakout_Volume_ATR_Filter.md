# Strategy: 6h_Camarilla_R4_S4_Breakout_Volume_ATR_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.505 | +7.0% | -6.1% | 480 | FAIL |
| ETHUSDT | 0.142 | +26.1% | -6.8% | 478 | PASS |
| SOLUSDT | -0.011 | +18.2% | -12.2% | 385 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.458 | +10.5% | -4.0% | 147 | PASS |

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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Camarilla: R4 = Close + ((High-Low) * 1.1/2), R3 = Close + ((High-Low) * 1.1/4)
    #          S3 = Close - ((High-Low) * 1.1/4), S4 = Close - ((High-Low) * 1.1/2)
    daily_range = daily_high - daily_low
    r4 = daily_close + (daily_range * 1.1 / 2)
    r3 = daily_close + (daily_range * 1.1 / 4)
    s3 = daily_close - (daily_range * 1.1 / 4)
    s4 = daily_close - (daily_range * 1.1 / 2)
    
    # Align HTF Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 6h price breaks above R4 with volume confirmation → long (strong continuation)
        # 2. 6h price breaks below S4 with volume confirmation → short (strong continuation)
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 6h breakout above R4 (strong continuation)
        if (close[i] > r4_6h[i] and            # 6h price above R4 Camarilla
            volume_ratio[i] > 1.3 and          # Volume confirmation
            atr_14[i] > 0.005 * close[i]):     # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below S4 (strong continuation)
        elif (close[i] < s4_6h[i] and          # 6h price below S4 Camarilla
              volume_ratio[i] > 1.3 and        # Volume confirmation
              atr_14[i] > 0.005 * close[i]):   # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-15 10:56
