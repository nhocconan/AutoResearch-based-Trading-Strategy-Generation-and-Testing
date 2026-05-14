# Strategy: 4h_CMF_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.006 | +20.3% | -15.6% | 219 | PASS |
| ETHUSDT | 0.230 | +32.3% | -14.2% | 203 | PASS |
| SOLUSDT | 0.533 | +69.6% | -29.0% | 192 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.395 | -7.1% | -9.5% | 80 | FAIL |
| ETHUSDT | 0.996 | +22.0% | -8.9% | 66 | PASS |
| SOLUSDT | -0.481 | -1.9% | -11.0% | 58 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Chaikin Money Flow (CMF) with 1d EMA trend filter and volume spike
# Long when CMF > 0 (buying pressure) + close > 1d EMA34 (uptrend) + volume spike
# Short when CMF < 0 (selling pressure) + close < 1d EAMA34 (downtrend) + volume spike
# Exit when CMF crosses zero or trend reverses
# CMF accumulates money flow volume, providing early signals of accumulation/distribution
# Designed for low trade frequency (~20-40/year) to minimize fee drain.
# Works in bull/bear by combining trend-following with CMF momentum and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Chaikin Money Flow (CMF) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    mfm = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0)
    
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    
    # 20-period CMF = sum of MFV over 20 periods / sum of Volume over 20 periods
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0.0)
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(cmf[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        cmf_val = cmf[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: CMF > 0 (buying pressure) + uptrend + volume spike
            if cmf_val > 0.0 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: CMF < 0 (selling pressure) + downtrend + volume spike
            elif cmf_val < 0.0 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: CMF crosses zero or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when CMF turns negative or trend turns down
                if cmf_val < 0.0 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when CMF turns positive or trend turns up
                if cmf_val > 0.0 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_CMF_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 02:22
