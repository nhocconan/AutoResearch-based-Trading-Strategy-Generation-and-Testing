# Strategy: 12h_Camarilla_R3S3_Breakout_VolumeSpike_1dEMA34_Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.274 | +31.6% | -6.5% | 87 | KEEP |
| ETHUSDT | 0.204 | +30.2% | -8.2% | 77 | KEEP |
| SOLUSDT | 0.151 | +27.6% | -24.0% | 75 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.974 | -2.7% | -7.1% | 35 | DISCARD |
| ETHUSDT | 0.386 | +11.3% | -6.4% | 29 | KEEP |
| SOLUSDT | -0.368 | +0.2% | -12.2% | 29 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with volume confirmation (>1.8x 20-bar volume MA) and 1d EMA34 trend filter
# Camarilla levels provide precise intraday support/resistance derived from prior day's range.
# Volume spike confirms institutional interest at key levels. 1d EMA34 ensures alignment with daily trend.
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull (breakouts with volume) and bear (failed reversals at strong levels reverse quickly).

name = "12h_Camarilla_R3S3_Breakout_VolumeSpike_1dEMA34_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Prior day's OHLC for Camarilla calculation (using 1d data)
    # Camarilla R3 = close + 1.1*(high-low)*1.1/4
    # Camarilla S3 = close - 1.1*(high-low)*1.1/4
    # Actually standard Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    prior_close = df_1d['close'].shift(1).values
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 34 for 1d EMA and 20 for volume MA, plus 1 for prior day shift
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Camarilla breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > camarilla_r3_aligned[i-1]  # Break above R3
        breakout_down = curr_close < camarilla_s3_aligned[i-1]  # Break below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up, price above 1d EMA34, volume spike
            if breakout_up and curr_close > ema_1d_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down, price below 1d EMA34, volume spike
            elif breakout_down and curr_close < ema_1d_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakdown or price below 1d EMA34
            if curr_close < camarilla_s3_aligned[i] or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout or price above 1d EMA34
            if curr_close > camarilla_r3_aligned[i] or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 18:01
