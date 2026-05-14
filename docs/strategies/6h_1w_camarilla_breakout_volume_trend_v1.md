# Strategy: 6h_1w_camarilla_breakout_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.086 | +23.5% | -3.5% | 133 | PASS |
| ETHUSDT | 0.423 | +31.6% | -5.4% | 109 | PASS |
| SOLUSDT | 0.731 | +65.7% | -12.0% | 89 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -3.120 | -4.2% | -4.7% | 43 | FAIL |
| ETHUSDT | 0.905 | +11.6% | -1.9% | 37 | PASS |
| SOLUSDT | -0.673 | +1.6% | -7.1% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Uses weekly Camarilla levels from daily data (R3, R4, S3, S4)
# - Long when price breaks above R4 with volume > 2x average AND 1w close > EMA20
# - Short when price breaks below S4 with volume > 2x average AND 1w close < EMA20
# - Exit when price retests R3/S3 or volume drops below average
# - Weekly trend filter ensures alignment with major trend
# - Volume confirmation prevents false breakouts
# - Targets 15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in both trending and ranging markets
# - Breakouts at R4/S4 have higher probability of continuation

name = "6h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute daily Camarilla pivots (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using previous day's data)
    camarilla_r4 = np.full_like(close_1d, np.nan)
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    camarilla_s4 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        if range_val > 0:
            camarilla_r4[i] = prev_close + range_val * 1.1 / 2
            camarilla_r3[i] = prev_close + range_val * 1.1 / 4
            camarilla_s3[i] = prev_close - range_val * 1.1 / 4
            camarilla_s4[i] = prev_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Pre-compute volume confirmation: > 2x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > R4 with volume spike AND 1w uptrend
            if (prices['close'].iloc[i] > r4_1d_aligned[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < S4 with volume spike AND 1w downtrend
            elif (prices['close'].iloc[i] < s4_1d_aligned[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests R3/S3 (mean reversion signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < r3_1d_aligned[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > s3_1d_aligned[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals
```

## Last Updated
2026-04-10 04:00
