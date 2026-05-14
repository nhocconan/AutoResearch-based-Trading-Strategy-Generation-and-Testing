# Strategy: 6h_WilliamsR_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.016 | +20.6% | -13.9% | 284 | KEEP |
| ETHUSDT | 0.485 | +50.2% | -14.7% | 257 | KEEP |
| SOLUSDT | 0.940 | +139.2% | -17.5% | 216 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.268 | -6.7% | -9.4% | 105 | DISCARD |
| ETHUSDT | 0.770 | +19.0% | -8.1% | 81 | KEEP |
| SOLUSDT | 0.117 | +7.1% | -9.8% | 76 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold reversal) in bull trend (close > 12h EMA50) with volume > 1.8x 20-period MA.
# Short when Williams %R crosses below -20 (overbought reversal) in bear trend (close < 12h EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. 12h EMA50 provides strong trend filter.
# Williams %R captures short-term exhaustion, ideal for 6h timeframe in ranging/volatile markets.
# Volume confirmation ensures institutional participation. Target: 75-150 total trades over 4 years (19-38/year).
# Works in both bull and bear markets: trend filter ensures we only trade in direction of 12h momentum,
# while Williams %R provides mean-reversion entries at extremes.

name = "6h_WilliamsR_12hEMA50_Volume"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Williams %R reversal conditions
        wr_oversold = wr > -80  # crossing above -80 from below
        wr_overbought = wr < -20  # crossing below -20 from above
        
        # Entry logic
        if position == 0:
            if is_bull_trend and wr_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and wr_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum loss) OR trend reversal
            if wr < -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum loss) OR trend reversal
            if wr > -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 07:33
