# Strategy: 4h_Bollinger_Bands_Breakout_ADX25_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.326 | +35.3% | -10.0% | 133 | PASS |
| ETHUSDT | 0.564 | +54.3% | -10.5% | 127 | PASS |
| SOLUSDT | 0.335 | +44.5% | -18.6% | 106 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.436 | -5.9% | -10.4% | 47 | FAIL |
| ETHUSDT | 0.556 | +13.7% | -9.0% | 43 | PASS |
| SOLUSDT | -0.460 | -1.6% | -17.8% | 36 | FAIL |

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
    
    # Hypothesis: 4h Bollinger Bands breakout with volume confirmation and ADX trend filter
    # Uses Bollinger Bands (20,2) for volatility-based breakout levels
    # ADX(14) > 25 ensures we trade in trending markets only
    # Volume surge (2x 20-period MA) confirms breakout strength
    # Works in bull/bear: breakouts from volatility bands with momentum capture moves
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # Calculate ADX components
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper band with volume spike and ADX > 25 (trending up)
            if close[i] > upper_band[i] and vol_spike[i] and adx[i] > 25 and plus_di[i] > minus_di[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band with volume spike and ADX > 25 (trending down)
            elif close[i] < lower_band[i] and vol_spike[i] and adx[i] > 25 and minus_di[i] > plus_di[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to middle band (SMA20) or opposite band touch
            if position == 1:
                if close[i] < sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Bollinger_Bands_Breakout_ADX25_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 06:12
