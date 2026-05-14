# Strategy: 12h_Camarilla_H3L3_VolumeSpike_1dEMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.293 | +10.8% | -9.1% | 78 | FAIL |
| ETHUSDT | 0.206 | +29.6% | -7.3% | 63 | PASS |
| SOLUSDT | 0.288 | +39.7% | -26.5% | 66 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.207 | +8.5% | -6.2% | 27 | PASS |
| SOLUSDT | -1.048 | -7.9% | -15.3% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Camarilla H3/L3 breakout with volume confirmation and 1d EMA34 trend filter.
- Long when price breaks above 1d Camarilla H3 level + volume > 2.0x 20-period 12h volume MA + price above 1d EMA34
- Short when price breaks below 1d Camarilla L3 level + volume > 2.0x 20-period 12h volume MA + price below 1d EMA34
- Fixed position size 0.25 to manage drawdown
- Uses Camarilla H3/L3 (moderate daily levels) + volume spike + 1d EMA trend
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
- Works in bull markets (buying breakouts with uptrend) and bear markets (selling breakdowns with downtrend)
"""

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
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Volume average (20-period) on 12h for confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla levels and EMA34 trend filter (HTF for structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate typical price for 1d
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels: H3/L3 = typical_price ± 1.1 * (high - low) / 4
    camarilla_h3_1d = typical_price_1d + 1.1 * (high_1d - low_1d) / 4.0
    camarilla_l3_1d = typical_price_1d - 1.1 * (high_1d - low_1d) / 4.0
    
    # Align all HTF indicators to primary timeframe (12h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        ema_34_val = ema_34_aligned[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 1d EMA34 trend filter
            # Long: price breaks above 1d Camarilla H3 + volume spike + price above 1d EMA34
            if price > camarilla_h3 and vol > 2.0 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 1d Camarilla L3 + volume spike + price below 1d EMA34
            elif price < camarilla_l3 and vol > 2.0 * vol_ma and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1d EMA34 (trend change) or opposite Camarilla level
            if price < ema_34_val or price < camarilla_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 1d EMA34 (trend change) or opposite Camarilla level
            if price > ema_34_val or price > camarilla_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_VolumeSpike_1dEMA34"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-17 21:45
