# Strategy: 4h_WilliamsAlligator_Trend_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.356 | +37.3% | -7.0% | 177 | PASS |
| ETHUSDT | 0.025 | +20.0% | -11.3% | 170 | PASS |
| SOLUSDT | 0.718 | +98.0% | -18.9% | 164 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.505 | +1.2% | -6.8% | 62 | FAIL |
| ETHUSDT | 0.192 | +8.4% | -8.8% | 61 | PASS |
| SOLUSDT | -0.290 | +0.5% | -11.6% | 61 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA trend filter with volume confirmation
# Uses the Alligator's jaw-teeth-lips alignment to identify trends
# Only trades when price is outside the Alligator's mouth (trending) and volume confirms
# Works in bull markets via teeth above jaw, in bear via teeth below jaw
# Target: 20-50 trades/year to avoid fee drag
name = "4h_WilliamsAlligator_Trend_1d_v1"
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
    
    # Get 1d data for multi-timeframe analysis (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter (stronger trend filter)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator (13,8,5 SMAs with 8,5,3 offsets)
    # Jaw: 13-period SMA, 8 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: 8-period SMA, 5 bars ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: 5-period SMA, 3 bars ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        # Alligator alignment checks
        # Bullish alignment: Lips > Teeth > Jaw (alligator opening up)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw (alligator opening down)
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish alignment + price above teeth + volume + 1d uptrend
            if bullish_alignment and price > teeth[i] and volume_filter and price > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price below teeth + volume + 1d downtrend
            elif bearish_alignment and price < teeth[i] and volume_filter and price < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bearish alignment or price crosses below jaw or ATR stop
            if bearish_alignment or price < jaw[i] or price < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bullish alignment or price crosses above jaw or ATR stop
            if bullish_alignment or price > jaw[i] or price > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 13:37
