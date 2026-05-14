# Strategy: 4h_12h_WilliamsAlligator_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.109 | +24.9% | -10.3% | 247 | PASS |
| ETHUSDT | 0.187 | +29.4% | -10.1% | 216 | PASS |
| SOLUSDT | 0.362 | +48.6% | -21.4% | 207 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.973 | -2.3% | -5.8% | 97 | FAIL |
| ETHUSDT | 0.555 | +13.8% | -7.3% | 92 | PASS |
| SOLUSDT | 0.193 | +8.3% | -8.2% | 70 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 12h EMA trend filter and volume confirmation.
# Alligator uses three SMAs (Jaw, Teeth, Lips) to identify trends and ranges.
# In strong trends, the SMAs diverge (mouth open); in ranges, they converge (mouth closed).
# Combined with 12h trend filter and volume spikes, it filters false signals.
# Target: 20-50 trades per year (80-200 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA(50) for 12h trend filter
    ema50_12h = np.zeros(len(close_12h))
    ema_multiplier = 2 / (50 + 1)
    ema50_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema50_12h[i] = (close_12h[i] - ema50_12h[i-1]) * ema_multiplier + ema50_12h[i-1]
    
    # Align 12h EMA to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Williams Alligator on 4h timeframe
    # Jaw: SMA(13, 8) - 13-period SMA shifted 8 bars forward
    # Teeth: SMA(8, 5) - 8-period SMA shifted 5 bars forward
    # Lips: SMA(5, 3) - 5-period SMA shifted 3 bars forward
    jaw = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    
    # Calculate SMAs
    def calculate_sma(data, period):
        sma = np.full(len(data), np.nan)
        if len(data) < period:
            return sma
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = sma[i-1] + (data[i] - data[i-period]) / period
        return sma
    
    sma13 = calculate_sma(close, 13)
    sma8 = calculate_sma(close, 8)
    sma5 = calculate_sma(close, 5)
    
    # Shift SMAs to create Alligator lines
    for i in range(8, n):
        jaw[i] = sma13[i-8] if i-8 >= 0 and not np.isnan(sma13[i-8]) else np.nan
    for i in range(5, n):
        teeth[i] = sma8[i-5] if i-5 >= 0 and not np.isnan(sma8[i-5]) else np.nan
    for i in range(3, n):
        lips[i] = sma5[i-3] if i-3 >= 0 and not np.isnan(sma5[i-3]) else np.nan
    
    # Average volume (20-period = 10 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_12h_aligned[i]
        
        # Alligator conditions
        # Mouth open (trending): Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        # Mouth closed (ranging): lines intertwined
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirm = vol > 1.8 * avg_vol
        
        if position == 0:
            # Long: Uptrend (Lips > Teeth > Jaw) + above 12h EMA50 + volume confirmation
            if (lips_val > teeth_val and teeth_val > jaw_val and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Downtrend (Lips < Teeth < Jaw) + below 12h EMA50 + volume confirmation
            elif (lips_val < teeth_val and teeth_val < jaw_val and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Trend changes to downtrend or price breaks below 12h EMA
            if (lips_val < teeth_val or teeth_val < jaw_val or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Trend changes to uptrend or price breaks above 12h EMA
            if (lips_val > teeth_val or teeth_val > jaw_val or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_WilliamsAlligator_Trend_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 21:49
