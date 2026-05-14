# Strategy: 4h_Camarilla_R3_S3_Breakout_1dADX_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.073 | +23.1% | -10.9% | 207 | PASS |
| ETHUSDT | 0.357 | +43.0% | -13.5% | 190 | PASS |
| SOLUSDT | 0.738 | +106.9% | -18.9% | 166 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.063 | -5.4% | -10.6% | 74 | FAIL |
| ETHUSDT | 0.318 | +10.7% | -12.3% | 68 | PASS |
| SOLUSDT | -0.689 | -6.7% | -20.5% | 56 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter
# Camarilla R3/S3 levels from 1d chart represent strong intraday support/resistance - breaks often lead to sustained moves
# 1d ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity with strong participation
# Discrete position sizing (0.25) controls fee drag while allowing meaningful exposure
# Target: 80-180 total trades over 4 years (20-45/year) for optimal risk-adjusted returns
# Works in bull markets by catching breakouts with trend, works in bear by only taking trend-aligned breaks
# Focus on BTC/ETH as primary symbols with SOL as secondary confirmation

name = "4h_Camarilla_R3_S3_Breakout_1dADX_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    # 1d data for ADX trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (trend strength filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate +DM and -DM
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[0], tr])
    
    # Smooth the values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(plus_dm) < period:
        return np.zeros(n)
    
    smoothed_plus_dm = wilders_smoothing(plus_dm, period)
    smoothed_minus_dm = wilders_smoothing(minus_dm, period)
    smoothed_tr = wilders_smoothing(tr, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d data for Camarilla pivot calculation
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3/S3 are strong breakout levels (not extreme like R4/S4)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    camarilla_r3 = pivot + (range_ * 1.1 / 4.0)   # R3 level
    camarilla_s3 = pivot - (range_ * 1.1 / 4.0)   # S3 level
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (need ADX > 25 for trending market)
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Camarilla R3 with volume confirmation and trending market
            if close[i] > camarilla_r3_aligned[i] and volume_confirmation[i] and trending:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 with volume confirmation and trending market
            elif close[i] < camarilla_s3_aligned[i] and volume_confirmation[i] and trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Camarilla S3 (reversal to downside) OR market becomes ranging (ADX < 20)
            if close[i] < camarilla_s3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Camarilla R3 (reversal to upside) OR market becomes ranging (ADX < 20)
            if close[i] > camarilla_r3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 04:48
