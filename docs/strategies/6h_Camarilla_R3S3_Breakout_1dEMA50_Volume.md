# Strategy: 6h_Camarilla_R3S3_Breakout_1dEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.138 | +26.4% | -11.8% | 118 | PASS |
| ETHUSDT | 0.230 | +32.4% | -13.6% | 98 | PASS |
| SOLUSDT | 0.815 | +117.1% | -21.4% | 87 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.034 | -4.3% | -10.0% | 43 | FAIL |
| ETHUSDT | 1.113 | +25.8% | -6.5% | 38 | PASS |
| SOLUSDT | 0.050 | +5.9% | -14.0% | 32 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 (camarilla resistance 3) in bull trend (close > 1d EMA50) with volume spike.
# Short when price breaks below S3 (camarilla support 3) in bear trend (close < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Camarilla levels derived from prior 1d candle (HLC) provide institutional pivot points.
# 1d EMA50 ensures alignment with higher timeframe trend. Volume confirmation reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_Volume"
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
    
    # Get 1d data for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 1d candle (HLC)
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R3 = CP + (H-L)*1.1/2, S3 = CP - (H-L)*1.1/2
    camarilla_pivot = typical_price.values
    camarilla_r3 = camarilla_pivot + (range_val * 1.1 / 2)
    camarilla_s3 = camarilla_pivot - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (use prior completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions
        breakout_long = close_val > r3_level
        breakout_short = close_val < s3_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_long and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_short and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 06:14
