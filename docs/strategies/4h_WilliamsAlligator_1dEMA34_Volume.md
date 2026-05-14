# Strategy: 4h_WilliamsAlligator_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.392 | +38.8% | -9.8% | 228 | PASS |
| ETHUSDT | 0.087 | +23.8% | -12.4% | 217 | PASS |
| SOLUSDT | 0.650 | +83.6% | -24.9% | 184 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.015 | -4.0% | -6.0% | 86 | FAIL |
| ETHUSDT | 1.327 | +28.4% | -6.6% | 69 | PASS |
| SOLUSDT | -0.151 | +3.1% | -10.2% | 59 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume spike
# Long when price is above Alligator's teeth (green line) in uptrend (close > 1d EMA34) with volume spike
# Short when price is below Alligator's teeth in downtrend (close < 1d EMA34) with volume spike
# Exit when price crosses the Alligator's jaw (red line) or trend reverses
# Williams Alligator: Jaw (blue) = SMA(13,8), Teeth (red) = SMA(8,5), Lips (green) = SMA(5,3)
# Uses median price (H+L)/2 as input. Designed for low trade frequency (~20-40/year) to minimize fee drain.
# Works in bull/bear by combining trend-following with Alligator's alignment and volume confirmation.

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
    
    # Calculate Williams Alligator on 4h data using median price
    high = prices['high'].values
    low = prices['low'].values
    median_price = (high + low) / 2.0
    
    # Jaw (blue line): 13-period SMA, smoothed by 8 periods
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().values
    
    # Teeth (red line): 8-period SMA, smoothed by 5 periods
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().values
    
    # Lips (green line): 5-period SMA, smoothed by 3 periods
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price above lips (green) + uptrend + volume spike
            if price > lips_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below lips (green) + downtrend + volume spike
            elif price < lips_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses jaw (blue line) or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below jaw or trend turns down
                if price < jaw_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above jaw or trend turns up
                if price > jaw_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 02:20
