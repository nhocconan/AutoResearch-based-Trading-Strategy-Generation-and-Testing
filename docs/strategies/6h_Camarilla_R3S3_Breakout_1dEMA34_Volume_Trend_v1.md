# Strategy: 6h_Camarilla_R3S3_Breakout_1dEMA34_Volume_Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.505 | +44.6% | -8.0% | 192 | PASS |
| ETHUSDT | 0.042 | +21.4% | -15.1% | 196 | PASS |
| SOLUSDT | 0.881 | +118.2% | -14.7% | 169 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.433 | -7.3% | -13.0% | 85 | FAIL |
| ETHUSDT | 1.122 | +23.7% | -6.0% | 60 | PASS |
| SOLUSDT | 0.020 | +5.6% | -10.6% | 60 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 6h timeframe for signal generation with Camarilla pivot breakouts
# 1d EMA(34) determines primary trend direction (bullish/bearish) - multi-timeframe alignment
# Volume confirmation (1.8x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Camarilla provides mathematical price levels based on prior day's range
# Volume confirms breakout validity, 1d EMA filter ensures trades only in higher timeframe trend direction
# Works in both bull and bear markets by only taking trades aligned with 1d trend

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Volume_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(34) for trend determination
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on prior day's high-low-close)
    # Camarilla formula: 
    # H4 = Close + 1.1*(High-Low)/2
    # L4 = Close - 1.1*(High-Low)/2
    # H3 = Close + 1.1*(High-Low)/4
    # L3 = Close - 1.1*(High-Low)/4
    # We use H3 as R3 resistance and L3 as S3 support
    hl_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * hl_range / 4
    camarilla_l3 = close_1d - 1.1 * hl_range / 4
    
    # Align Camarilla levels to 6h timeframe (use prior day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla H3 (R3) + volume confirm + price > 1d EMA34 (bullish trend)
            if close[i] > camarilla_h3_aligned[i] and volume_confirm[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla L3 (S3) + volume confirm + price < 1d EMA34 (bearish trend)
            elif close[i] < camarilla_l3_aligned[i] and volume_confirm[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla L3 (S3) or price < 1d EMA34 (trend reversal)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla H3 (R3) or price > 1d EMA34 (trend reversal)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 20:52
