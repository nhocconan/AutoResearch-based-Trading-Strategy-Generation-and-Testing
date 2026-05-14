# Strategy: 6h_Camarilla_R4_S4_Breakout_12hADX_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.135 | +26.4% | -10.7% | 194 | PASS |
| ETHUSDT | 0.256 | +34.5% | -13.1% | 174 | PASS |
| SOLUSDT | 0.491 | +65.2% | -16.6% | 144 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.812 | -11.9% | -16.1% | 75 | FAIL |
| ETHUSDT | 0.965 | +22.8% | -7.9% | 53 | PASS |
| SOLUSDT | -0.571 | -4.2% | -15.8% | 53 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 12h ADX trend filter and volume confirmation
# Camarilla R4/S4 levels from 12h chart represent strong breakout zones - breaks often lead to sustained moves
# 12h ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions
# Volume spike (>1.8 x 24-period EMA) confirms breakout validity
# Discrete position sizing (0.25) controls fee drag while allowing meaningful exposure
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns
# Works in bull markets by catching breakouts with trend, works in bear by only taking trend-aligned breaks

name = "6h_Camarilla_R4_S4_Breakout_12hADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 1.8 x 24-period EMA)
    vol_ema_24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_24)
    
    # 12h data for ADX trend filter and Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # 12h ADX calculation (trend strength filter)
    # ADX requires +DI, -DI, and DX calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate +DM and -DM
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Calculate True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[0], tr])
    
    # Smooth the values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
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
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # 12h data for Camarilla pivot calculation
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Camarilla levels: R4/S4 are the extreme levels for significant breakouts
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    camarilla_r4 = pivot + (range_ * 1.1 / 2.0)   # R4 level
    camarilla_s4 = pivot - (range_ * 1.1 / 2.0)   # S4 level
    
    # Align Camarilla levels to 6h timeframe (wait for completed 12h bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h ADX (need ADX > 25 for trending market)
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Camarilla R4 with volume confirmation and trending market
            if close[i] > camarilla_r4_aligned[i] and volume_confirmation[i] and trending:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S4 with volume confirmation and trending market
            elif close[i] < camarilla_s4_aligned[i] and volume_confirmation[i] and trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Camarilla S4 (reversal to downside) OR market becomes ranging (ADX < 20)
            if close[i] < camarilla_s4_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Camarilla R4 (reversal to upside) OR market becomes ranging (ADX < 20)
            if close[i] > camarilla_r4_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 04:45
