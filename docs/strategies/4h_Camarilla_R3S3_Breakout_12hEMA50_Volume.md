# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.186 | +28.9% | -11.7% | 239 | PASS |
| ETHUSDT | 0.293 | +36.1% | -12.5% | 229 | PASS |
| SOLUSDT | 0.242 | +36.4% | -29.2% | 186 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.781 | -2.1% | -6.2% | 88 | FAIL |
| ETHUSDT | 1.637 | +35.9% | -6.4% | 80 | PASS |
| SOLUSDT | 0.477 | +13.0% | -9.7% | 66 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R3 (bullish breakout level) in bull trend (close > 12h EMA50) with volume > 2x 20-period MA.
# Short when price breaks below S3 (bearish breakdown level) in bear trend (close < 12h EMA50) with volume spike.
# Uses discrete position sizing (0.30) to minimize fee churn while maintaining sufficient exposure.
# 12h EMA50 provides intermediate-term trend filter to avoid counter-trend trades in both bull and bear markets.
# Volume confirmation ensures breakouts have institutional participation, reducing false signals.
# Target: 100-200 total trades over 4 years (25-50/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Volume"
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
    
    # Get 12h data for EMA50 trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior 12h candle (HLC)
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    range_val = df_12h['high'] - df_12h['low']
    
    camarilla_pivot = typical_price.values
    # Camarilla R3 = CP + (H-L)*1.1/4, S3 = CP - (H-L)*1.1/4
    camarilla_r3 = camarilla_pivot + (range_val * 1.1 / 4)
    camarilla_s3 = camarilla_pivot - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (use prior completed 12h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions at R3/S3
        breakout_long = close_val > r3_level
        breakout_short = close_val < s3_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_long and vol_spike:
                signals[i] = 0.30
                position = 1
            elif is_bear_trend and breakout_short and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above R3 OR trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-03 06:19
