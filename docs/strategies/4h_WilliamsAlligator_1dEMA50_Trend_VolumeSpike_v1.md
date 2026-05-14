# Strategy: 4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.380 | +37.2% | -9.1% | 181 | PASS |
| ETHUSDT | 0.273 | +34.0% | -17.4% | 160 | PASS |
| SOLUSDT | 0.650 | +83.4% | -18.8% | 142 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.302 | +3.1% | -5.2% | 68 | FAIL |
| ETHUSDT | 0.779 | +18.0% | -7.4% | 60 | PASS |
| SOLUSDT | 0.144 | +7.6% | -7.2% | 49 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator identifies trend phases via three SMAs (Jaw/Teeth/Lips); trades only when aligned
# 1d EMA50 ensures higher timeframe trend alignment; volume spike confirms institutional participation
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag and avoid overtrading
# Works in both bull/bear markets by trading with the 1d trend direction

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components: SMAs of median price (typical price)
    typical_price = (high + low + close) / 3.0
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().values     # 5-period SMA
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20, 13)  # Need sufficient history for 1d EMA, volume MA, and Alligator
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals: aligned (bullish) or reversed (bearish)
        bullish_aligned = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])  # Lips > Teeth > Jaw
        bearish_aligned = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])  # Lips < Teeth < Jaw
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment, volume spike, uptrend
            if bullish_aligned and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment, volume spike, downtrend
            elif bearish_aligned and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish alignment or trend reversal
            if bearish_aligned or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish alignment or trend reversal
            if bullish_aligned or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 19:29
