# Strategy: 6h_WilliamsR_1dEMA34_VolumeSpike_MR

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.159 | +27.4% | -13.9% | 271 | PASS |
| ETHUSDT | 0.219 | +31.6% | -16.5% | 255 | PASS |
| SOLUSDT | 0.911 | +132.9% | -17.5% | 215 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.226 | -6.1% | -8.4% | 104 | FAIL |
| ETHUSDT | 0.644 | +16.4% | -7.5% | 79 | PASS |
| SOLUSDT | 0.408 | +12.1% | -7.2% | 74 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold reversal) in bull trend (close > 1d EMA34) with volume spike.
# Short when Williams %R crosses below -20 (overbought reversal) in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Williams %R is effective for mean reversion in ranging markets and captures reversals in trends.
# The 1d EMA34 filter ensures we only take mean-reversion trades in the direction of the higher timeframe trend.
# Volume confirmation reduces false signals. Designed for 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_1dEMA34_VolumeSpike_MR"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need at least 34 for EMA + 1 for current
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R on 6h timeframe (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        wr = williams_r[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Williams %R reversal conditions
        wr_long_signal = wr > -80  # Crossed above oversold
        wr_short_signal = wr < -20  # Crossed below overbought
        
        # Entry logic
        if position == 0:
            if is_bull_trend and wr_long_signal and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and wr_short_signal and vol_spike:
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
2026-05-03 06:08
