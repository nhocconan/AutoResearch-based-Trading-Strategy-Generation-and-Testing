# Strategy: 4h_Donchian20_12hEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.006 | +19.1% | -12.3% | 76 | PASS |
| ETHUSDT | 0.020 | +18.1% | -16.6% | 72 | PASS |
| SOLUSDT | 0.712 | +116.3% | -28.1% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.783 | -2.2% | -9.6% | 25 | FAIL |
| ETHUSDT | 0.305 | +10.8% | -8.6% | 26 | PASS |
| SOLUSDT | -0.366 | -2.6% | -21.3% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(34) trend filter and volume spike confirmation.
# Uses Donchian channel (20-period high/low) as price channel structure.
# Long when price breaks above upper Donchian band with bullish 12h trend and volume spike.
# Short when price breaks below lower Donchian band with bearish 12h trend and volume spike.
# Uses 12h EMA(34) as higher timeframe trend filter to avoid counter-trend trades.
# Designed for 4h timeframe to target 20-50 trades/year per symbol.
# Works in bull/bear via multi-timeframe trend alignment and volatility-based signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian calculation and trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian channel (20-period) on 12h data
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    # Align Donchian levels to 4h timeframe (using previous 12h bar's values)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 12h EMA(34) for higher timeframe trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike filter (20-period on 4h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer, higher quality trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band + bullish 12h trend + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + bearish 12h trend + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on break below lower Donchian band or trend reversal
                if (close[i] < donchian_low_aligned[i] or 
                    close[i] < ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on break above upper Donchian band or trend reversal
                if (close[i] > donchian_high_aligned[i] or 
                    close[i] > ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 09:25
