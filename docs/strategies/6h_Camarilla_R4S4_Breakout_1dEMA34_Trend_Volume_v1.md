# Strategy: 6h_Camarilla_R4S4_Breakout_1dEMA34_Trend_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.022 | +20.3% | -13.2% | 63 | KEEP |
| ETHUSDT | 0.413 | +47.8% | -11.9% | 59 | KEEP |
| SOLUSDT | 0.678 | +103.9% | -25.4% | 53 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.266 | -7.8% | -13.0% | 28 | DISCARD |
| ETHUSDT | 0.095 | +6.7% | -7.1% | 23 | KEEP |
| SOLUSDT | -0.520 | -5.1% | -20.7% | 19 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1d EMA34 trend and volume confirmation
# Camarilla R4/S4 levels represent stronger breakout thresholds than R3/S3.
# Breakouts above R4 or below S4 with 1d EMA34 trend alignment capture strong momentum.
# Volume confirmation (2.0x 20-period average) filters false breakouts.
# Works in both bull/bear markets by only taking breakouts aligned with 1d EMA34.
# Discrete sizing 0.25 targets ~50-100 trades over 4 years (12-25/year).

name = "6h_Camarilla_R4S4_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for EMA34 trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    camarilla_r4 = prev_1d_close + 1.5 * (prev_1d_high - prev_1d_low)
    camarilla_s4 = prev_1d_close - 1.5 * (prev_1d_high - prev_1d_low)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R4 with 1d uptrend (close > EMA34)
            long_breakout = close[i] > camarilla_r4_aligned[i]
            # Short breakdown: price < S4 with 1d downtrend (close < EMA34)
            short_breakout = close[i] < camarilla_s4_aligned[i]
            
            # 1d EMA34 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_34_1d_aligned[i]
            ema_trend_down = close[i] < ema_34_1d_aligned[i]
            
            if long_breakout and ema_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and ema_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < S4 or trend reversal (close < EMA34)
            if close[i] < camarilla_s4_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > R4 or trend reversal (close > EMA34)
            if close[i] > camarilla_r4_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 23:28
