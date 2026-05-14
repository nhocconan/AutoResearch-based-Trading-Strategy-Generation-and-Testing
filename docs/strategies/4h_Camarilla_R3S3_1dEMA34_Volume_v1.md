# Strategy: 4h_Camarilla_R3S3_1dEMA34_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.310 | +29.8% | -8.8% | 297 | PASS |
| ETHUSDT | 0.121 | +25.0% | -9.3% | 275 | PASS |
| SOLUSDT | -0.052 | +16.7% | -11.7% | 234 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.131 | -0.9% | -4.8% | 121 | FAIL |
| ETHUSDT | 1.960 | +27.8% | -2.6% | 98 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 1d EMA34 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below S3 AND close < 1d EMA34 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price retouches the Camarilla pivot level (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide high-probability reversal points; EMA34 ensures higher-timeframe trend alignment
# Volume spike confirms institutional participation; pivot retouch exit works in ranging markets
# Specifically designed for 4h timeframe to reduce trade frequency vs lower timeframes while capturing significant moves
# Works in both bull and bear markets: trend filter captures directional moves, mean reversion exit works in ranges

name = "4h_Camarilla_R3S3_1dEMA34_Volume_v1"
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
    
    # Calculate Camarilla levels for 4h timeframe
    # Based on previous bar's high, low, close
    ph = np.concatenate([[high[0]], high[:-1]])  # previous high
    pl = np.concatenate([[low[0]], low[:-1]])    # previous low
    pc = np.concatenate([[close[0]], close[:-1]]) # previous close
    
    pivot = (ph + pl + pc) / 3.0
    range_ = ph - pl
    
    # Camarilla levels
    r3 = pivot + (range_ * 1.1 / 4.0)
    s3 = pivot - (range_ * 1.1 / 4.0)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: Break above R3 AND uptrend AND volume spike
            if close[i] > r3[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 AND downtrend AND volume spike
            elif close[i] < s3[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retouches pivot level (mean reversion)
            if close[i] <= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retouches pivot level (mean reversion)
            if close[i] >= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 11:53
