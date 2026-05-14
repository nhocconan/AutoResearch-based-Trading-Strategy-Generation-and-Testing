# Strategy: 4h_Camarilla_R3_S3_1dEMA34_VolumeSpike_ATRStop_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.174 | +28.0% | -9.9% | 165 | PASS |
| ETHUSDT | 0.192 | +29.9% | -11.5% | 154 | PASS |
| SOLUSDT | 0.924 | +124.0% | -16.6% | 135 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.057 | +6.5% | -5.7% | 60 | PASS |
| ETHUSDT | 1.094 | +23.2% | -6.7% | 57 | PASS |
| SOLUSDT | 0.448 | +12.2% | -11.2% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses ATR-based trailing stop for risk management. Discrete sizing 0.25.
# Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla levels provide high-probability reversal/breakout points from prior 1d range.
# 1d EMA34 filter ensures alignment with daily trend to avoid counter-trend trades.
# Volume spike confirms institutional participation at key levels.
# Based on proven Camarilla patterns showing strong test performance in DB (ETH/SOL winners).

name = "4h_Camarilla_R3_S3_1dEMA34_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d OHLC for Camarilla pivot levels (from prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 completed bar for prior
        return np.zeros(n)
    
    # Use prior completed 1d bar's OHLC for Camarilla calculation
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_open = np.roll(df_1d['open'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    prior_open[0] = np.nan
    
    # Calculate Camarilla levels for prior 1d bar
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    cam_high = prior_close + (prior_high - prior_low) * 1.1 / 2
    cam_low = prior_close - (prior_high - prior_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    cam_high_aligned = align_htf_to_ltf(prices, df_1d, cam_high)
    cam_low_aligned = align_htf_to_ltf(prices, df_1d, cam_low)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(30) for stoploss (using 4h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=30, min_periods=30, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 30-bar average (on 4h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        upper = cam_high_aligned[i]
        lower = cam_low_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(upper) or np.isnan(lower) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above Camarilla R3 with volume spike and above 1d EMA34
        long_entry = (close[i] > upper) and (close[i] > ema_trend) and vol_spike
        # Short: break below Camarilla S3 with volume spike and below 1d EMA34
        short_entry = (close[i] < lower) and (close[i] < ema_trend) and vol_spike
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 03:47
