# Strategy: 4h_Camarilla_R3S3_Breakout_12hHMA21_4hVolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.340 | +0.9% | -18.5% | 180 | FAIL |
| ETHUSDT | 0.220 | +32.9% | -14.1% | 157 | PASS |
| SOLUSDT | 0.666 | +99.4% | -22.3% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.976 | +24.2% | -8.8% | 52 | PASS |
| SOLUSDT | 0.071 | +6.3% | -12.4% | 46 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h HMA21 trend filter and 4h volume spike confirmation.
# Long when price breaks above R3 with price > 12h HMA21 (bullish trend) and 4h volume > 2.0x 20-period average.
# Short when price breaks below S3 with price < 12h HMA21 (bearish trend) and 4h volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts).
# Uses 4h timeframe for balance of trade frequency and cost, 12h HMA21 for smooth trend filter, volume spike for confirmation.
# Target: 75-200 total trades over 4 years (19-50/year).
# HMA21 provides stronger trend filter than EMA50, reducing whipsaws in sideways markets.

name = "4h_Camarilla_R3S3_Breakout_12hHMA21_4hVolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    # Final WMA of sqrt period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

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
    # 4h volume confirmation: > 2.0x 20-period average (stricter filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (2.0 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h HMA(21) - trend filter
    hma_21 = calculate_hma(close_12h, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
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
            camarilla_r3[i] = close_val + (range_val * 1.1 / 2)  # R3
            camarilla_s3[i] = close_val - (range_val * 1.1 / 2)  # S3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(hma_21_aligned[i]) or
            np.isnan(volume_spike_4h[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + price > 12h HMA21 (bullish) + 4h volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > hma_21_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price < 12h HMA21 (bearish) + 4h volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < hma_21_aligned[i] and 
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
2026-05-14 01:58
