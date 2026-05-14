# Strategy: 4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.556 | +46.3% | -9.8% | 247 | KEEP |
| ETHUSDT | 0.310 | +36.4% | -10.4% | 236 | KEEP |
| SOLUSDT | 0.699 | +87.6% | -23.1% | 214 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.979 | -2.7% | -7.3% | 101 | DISCARD |
| ETHUSDT | 1.094 | +22.8% | -5.8% | 89 | KEEP |
| SOLUSDT | 0.841 | +18.2% | -9.3% | 71 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2
Hypothesis: On 4h timeframe, Camarilla R1/S1 levels from 1d act as strong intraday support/resistance. 
A break above R1 with volume spike and 1d uptrend (price > EMA34) signals long; break below S1 with 
volume spike and 1d downtrend (price < EMA34) signals short. Uses ATR-based stoploss and discrete 
position sizing (0.30) to limit trades (~25-35/year) and fee drag. Designed to work in both bull 
and bear markets by trading institutional levels with trend and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla levels and trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla R1/S1 levels from previous 1d bar
    # Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF (4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h ATR for volatility and stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    max_high = 0.0     # track highest high since entry for trailing stop (long)
    min_low = 0.0      # track lowest low since entry for trailing stop (short)
    
    # Start index: need ATR (14), volume MA (20) + aligned HTF arrays
    start_idx = max(20, 14, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above camarilla R1 with volume spike and 1d uptrend
            long_breakout = (curr_close > camarilla_r1_aligned[i]) and vol_spike[i] and (curr_close > ema_34_1d_aligned[i])
            # Short: price breaks below camarilla S1 with volume spike and 1d downtrend
            short_breakout = (curr_close < camarilla_s1_aligned[i]) and vol_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                max_high = curr_high
            elif short_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                min_low = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            max_high = max(max_high, curr_high)
            # Exit: price breaks below camarilla S1 OR trend turns down OR ATR trailing stop hit
            trailing_stop = curr_high < (max_high - 2.0 * atr_14[i])
            if (curr_close < camarilla_s1_aligned[i]) or (curr_close < ema_34_1d_aligned[i]) or trailing_stop:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            min_low = min(min_low, curr_low)
            # Exit: price breaks above camarilla R1 OR trend turns up OR ATR trailing stop hit
            trailing_stop = curr_low > (min_low + 2.0 * atr_14[i])
            if (curr_close > camarilla_r1_aligned[i]) or (curr_close > ema_34_1d_aligned[i]) or trailing_stop:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 12:01
