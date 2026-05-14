# Strategy: 6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.396 | +40.1% | -10.8% | 115 | PASS |
| ETHUSDT | 0.214 | +31.4% | -12.8% | 111 | PASS |
| SOLUSDT | 0.660 | +89.2% | -18.9% | 85 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.989 | -4.5% | -8.3% | 47 | FAIL |
| ETHUSDT | 1.291 | +29.0% | -6.4% | 33 | PASS |
| SOLUSDT | -0.277 | +0.8% | -14.2% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation.
# Uses Camarilla pivot levels calculated from daily high/low/close.
# Long when price breaks above R3 with bullish 1d trend and volume spike.
# Short when price breaks below S3 with bearish 1d trend and volume spike.
# Uses 1d EMA(34) as higher timeframe trend filter to avoid counter-trend trades.
# Designed for 6h timeframe to target 12-37 trades/year per symbol.
# Works in bull/bear via multi-timeframe trend alignment and volatility-based signals.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivots and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    # R4 = C + (H - L) * 1.1
    # S4 = C - (H - L) * 1.1
    
    # Calculate for each day, then align to 6h
    typical_price = (high_1d + low_1d + close_1d) / 3
    price_range = high_1d - low_1d
    
    camarilla_pivot = typical_price
    camarilla_r3 = close_1d + price_range * 1.1 / 2
    camarilla_s3 = close_1d - price_range * 1.1 / 2
    camarilla_r4 = close_1d + price_range * 1.1
    camarilla_s4 = close_1d - price_range * 1.1
    
    # Align Camarilla levels to 6s timeframe (using previous day's values)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer, higher quality trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + bullish 1d trend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + bearish 1d trend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on break below S3 or trend reversal
                if (close[i] < camarilla_s3_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on break above R3 or trend reversal
                if (close[i] > camarilla_r3_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 09:25
