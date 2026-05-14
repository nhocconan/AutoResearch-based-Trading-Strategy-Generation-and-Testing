# Strategy: 6h_12hCamarillaR4S4_1dEMA34Trend_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.275 | +32.1% | -9.3% | 51 | PASS |
| ETHUSDT | 0.175 | +28.5% | -12.9% | 45 | PASS |
| SOLUSDT | 1.114 | +133.3% | -14.3% | 40 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.166 | +4.5% | -4.8% | 16 | FAIL |
| ETHUSDT | 0.273 | +9.5% | -11.0% | 16 | PASS |
| SOLUSDT | -0.370 | -0.4% | -16.6% | 13 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R4 level AND 1d EMA34 > EMA200 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 12h Camarilla S4 level AND 1d EMA34 < EMA200 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 12h EMA34 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 12h Camarilla provides clear structure with proven breakout/fade edge
# 1d EMA34/EMA200 filter ensures alignment with higher timeframe trend (works in bull/bear)
# Volume confirmation filters weak breakouts (reduces false signals)

name = "6h_12hCamarillaR4S4_1dEMA34Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla pivots and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels based on previous 12h bar
    # Camarilla: R4 = Close + 1.5 * (High - Low), S4 = Close - 1.5 * (High - Low)
    camarilla_r4_12h = close_12h + 1.5 * (high_12h - low_12h)
    camarilla_s4_12h = close_12h - 1.5 * (high_12h - low_12h)
    
    # Calculate 12h EMA34 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema_34_12h = close_series_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h Camarilla levels and EMA to 6h timeframe (wait for completed 12h bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4_12h)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4_12h)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Align 1d EMA indicators to 6h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R4 with 1d EMA34 > EMA200 and volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and close[i-1] <= camarilla_r4_aligned[i-1] and 
                ema_34_1d_aligned[i] > ema_200_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S4 with 1d EMA34 < EMA200 and volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and close[i-1] >= camarilla_s4_aligned[i-1] and 
                  ema_34_1d_aligned[i] < ema_200_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA34 (trend reversal)
            if close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA34 (trend reversal)
            if close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 07:56
