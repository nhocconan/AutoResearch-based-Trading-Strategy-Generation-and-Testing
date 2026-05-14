# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_4hVolumeSpike_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.353 | +36.2% | -7.1% | 194 | PASS |
| ETHUSDT | 0.055 | +22.1% | -14.1% | 187 | PASS |
| SOLUSDT | 0.791 | +101.4% | -18.7% | 166 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.869 | -1.7% | -7.8% | 78 | FAIL |
| ETHUSDT | 0.723 | +16.8% | -11.5% | 61 | PASS |
| SOLUSDT | -0.194 | +2.6% | -9.9% | 54 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 4h volume spike confirmation.
# Long when price breaks above R3 with price > 1d EMA34 (bullish trend) and 4h volume > 2.0x 20-period average.
# Short when price breaks below S3 with price < 1d EMA34 (bearish trend) and 4h volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts).
# Uses 4h timeframe for higher frequency than 12h but with tight entry conditions to avoid overtrading.
# Volume spike confirmation (2.0x) reduces false breakouts. Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_4hVolumeSpike_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter (standard period for strong trend signal)
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # --- 4h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Precompute prior day's OHLC for each 4h bar using vectorized approach
    open_time = prices['open_time']
    prior_day_start = open_time - pd.Timedelta(days=1)
    prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
    
    # Merge to get prior day's OHLC for each timestamp
    df_1d_pivot = df_1d_pivot.copy()
    df_1d_pivot['date'] = df_1d_pivot['open_time'].dt.date
    prior_day_start_date = prior_day_start.dt.date
    
    # Create mapping from date to OHLC
    ohlc_map = df_1d_pivot.groupby('date').agg({
        'high': 'first',
        'low': 'first',
        'close': 'first'
    })
    
    for i in range(n):
        pd_date = prior_day_start_date.iloc[i]
        if pd_date in ohlc_map.index:
            day_data = ohlc_map.loc[pd_date]
            high_val = day_data['high']
            low_val = day_data['low']
            close_val = day_data['close']
            range_val = high_val - low_val
            camarilla_r3[i] = close_val + (range_val * 1.1 / 4)  # R3
            camarilla_s3[i] = close_val - (range_val * 1.1 / 4)  # S3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike_4h[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + price > 1d EMA34 (bullish) + 4h volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price < 1d EMA34 (bearish) + 4h volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-14 02:02
