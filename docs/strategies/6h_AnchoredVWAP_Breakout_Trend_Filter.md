# Strategy: 6h_AnchoredVWAP_Breakout_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.066 | +22.7% | -11.6% | 44 | PASS |
| ETHUSDT | 0.062 | +22.1% | -15.0% | 50 | PASS |
| SOLUSDT | 1.013 | +166.8% | -20.9% | 43 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.960 | -0.5% | -5.7% | 17 | FAIL |
| ETHUSDT | 0.186 | +8.3% | -6.5% | 12 | PASS |
| SOLUSDT | -0.606 | -2.6% | -9.5% | 15 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_AnchoredVWAP_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for anchor VWAP and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate anchored VWAP from start of each daily candle
    # VWAP = cumulative(close * volume) / cumulative(volume)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    cum_vol_price = np.cumsum(close_1d * volume_1d)
    cum_vol = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_vol_price, cum_vol, out=np.full_like(cum_vol_price, np.nan), where=cum_vol!=0)
    
    # Align VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Daily EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike detection (6h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Require strong volume spike
        
        if position == 0:
            # Long: Price breaks above anchored VWAP with daily uptrend and volume spike
            if close[i] > vwap_1d_aligned[i] and close[i] > ema_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below anchored VWAP with daily downtrend and volume spike
            elif close[i] < vwap_1d_aligned[i] and close[i] < ema_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below anchored VWAP or trend turns down
            if close[i] < vwap_1d_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above anchored VWAP or trend turns up
            if close[i] > vwap_1d_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 10:43
