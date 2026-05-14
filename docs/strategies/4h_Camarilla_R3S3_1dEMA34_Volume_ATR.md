# Strategy: 4h_Camarilla_R3S3_1dEMA34_Volume_ATR

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.085 | +17.8% | -9.3% | 99 | FAIL |
| ETHUSDT | 0.555 | +46.9% | -7.7% | 78 | PASS |
| SOLUSDT | 0.662 | +71.2% | -11.8% | 74 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.419 | +10.9% | -6.0% | 32 | PASS |
| SOLUSDT | -0.474 | +0.3% | -6.0% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 1d EMA34 for strong trend direction (long only when price > EMA34, short only when price < EMA34).
# Entry: price breaks above Camarilla R3 level with volume > 2.0x 20-period MA for longs,
#        or breaks below Camarilla S3 level with volume spike for shorts.
# Exit: ATR(14) trailing stop (2.5x ATR) or reversal of 1d EMA34 trend.
# Discrete sizing 0.25. Target: 80-180 total trades over 4 years (20-45/year).
# Camarilla levels from 1d provide robust daily support/resistance; 1d EMA34 filters counter-trend trades;
# volume confirmation reduces false breakouts. Works in bull via trend-following breakouts
# and in bear via short breakdowns with trend alignment.

name = "4h_Camarilla_R3S3_1dEMA34_Volume_ATR"
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
    
    # Get 1d data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[tr[0]], tr])  # same length as prices
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R3 = close + 1.1666*(high-low), S3 = close - 1.1666*(high-low)
    camarilla_r3_1d = df_1d['close'] + 1.1666 * (df_1d['high'] - df_1d['low'])
    camarilla_s3_1d = df_1d['close'] - 1.1666 * (df_1d['high'] - df_1d['low'])
    # Align to 4h timeframe (wait for 1d bar to close)
    camarilla_r3 = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d.values)
    camarilla_s3 = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d.values)
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long ATR stop
    lowest_since_entry = 0.0   # for short ATR stop
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_r3[i]
        s3_level = camarilla_s3[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Update highest/lowest since entry for ATR stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i]) if lowest_since_entry != 0 else low[i]
        
        # Entry logic
        if position == 0:
            # Long: break above R3 with volume spike in uptrend
            if close_val > r3_level and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = high[i]
            # Short: break below S3 with volume spike in downtrend
            elif close_val < s3_level and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = low[i]
        elif position == 1:
            # Long exit: ATR stoploss OR price breaks below S3 OR trend turns down
            atr_stop = highest_since_entry - (2.5 * atr_val)
            if close_val < atr_stop or close_val < s3_level or not is_uptrend:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ATR stoploss OR price breaks above R3 OR trend turns up
            atr_stop = lowest_since_entry + (2.5 * atr_val)
            if close_val > atr_stop or close_val > r3_level or not is_downtrend:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 08:21
