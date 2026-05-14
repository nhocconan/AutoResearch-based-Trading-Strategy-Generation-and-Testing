# Strategy: 6h_Camarilla_R3S3_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.521 | +50.9% | -10.0% | 77 | PASS |
| ETHUSDT | 0.422 | +47.6% | -13.5% | 73 | PASS |
| SOLUSDT | 0.699 | +107.4% | -28.8% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.364 | -8.5% | -12.9% | 34 | FAIL |
| ETHUSDT | 0.007 | +5.1% | -9.1% | 31 | PASS |
| SOLUSDT | -0.524 | -5.1% | -20.7% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long: Close breaks above R3 AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period MA
# Short: Close breaks below S3 AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period MA
# Exit: Opposite Camarilla breakout or EMA34 trend reversal.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla levels provide strong intraday support/resistance; 1d EMA34 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals with trend alignment
# and in bear via short signals with trend alignment.

name = "6h_Camarilla_R3S3_1dEMA34_Volume"
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
    open_price = prices['open'].values  # needed for Camarilla calculation
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels using previous day's OHLC (standard formula)
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    # Using previous 1d bar's OHLC (already completed)
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    R3 = prev_1d_close + 1.1 * (prev_1d_high - prev_1d_low)
    S3 = prev_1d_close - 1.1 * (prev_1d_high - prev_1d_low)
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above R3 AND uptrend AND volume spike
            if close_val > R3_aligned[i] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 AND downtrend AND volume spike
            elif close_val < S3_aligned[i] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below S3 OR trend turns down
            if close_val < S3_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above R3 OR trend turns up
            if close_val > R3_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 08:37
