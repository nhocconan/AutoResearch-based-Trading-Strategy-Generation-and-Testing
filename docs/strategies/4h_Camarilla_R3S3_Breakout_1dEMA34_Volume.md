# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.385 | +41.1% | -11.4% | 192 | KEEP |
| ETHUSDT | 0.125 | +26.0% | -14.6% | 188 | KEEP |
| SOLUSDT | 0.821 | +128.1% | -26.9% | 163 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.014 | -5.9% | -9.1% | 75 | DISCARD |
| ETHUSDT | 1.382 | +34.8% | -11.8% | 60 | KEEP |
| SOLUSDT | -0.409 | -1.9% | -11.7% | 55 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 (bullish breakout level) in bull trend (close > 1d EMA34) with volume > 2x 20-period MA.
# Short when price breaks below S3 (bearish breakdown level) in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.30) to minimize fee churn while maintaining sufficient exposure.
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades in both bull and bear markets.
# Volume confirmation ensures breakouts have institutional participation, reducing false signals.
# Target: 100-200 total trades over 4 years (25-50/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d candle (HLC)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    camarilla_pivot = typical_price.values
    # Camarilla R3 = CP + (H-L)*1.1/4, S3 = CP - (H-L)*1.1/4
    camarilla_r3 = camarilla_pivot + (range_val * 1.1 / 4)
    camarilla_s3 = camarilla_pivot - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (use prior completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
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
2026-05-03 07:33
