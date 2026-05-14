# Strategy: 4h_Camarilla_R3S3_1dEMA34_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.036 | +21.7% | -11.0% | 254 | PASS |
| ETHUSDT | 0.109 | +25.1% | -12.5% | 231 | PASS |
| SOLUSDT | 0.642 | +82.9% | -21.2% | 208 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.075 | -3.8% | -6.1% | 90 | FAIL |
| ETHUSDT | 1.459 | +29.3% | -9.2% | 79 | PASS |
| SOLUSDT | 0.613 | +15.2% | -9.0% | 70 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 + 1d EMA34 uptrend + volume > 1.8x 20-bar average.
# Short when price breaks below S3 + 1d EMA34 downtrend + volume > 1.8x 20-bar average.
# ATR trailing stop (2.0x) for risk management.
# Targets 75-200 total trades over 4 years (19-50/year) with discrete position sizing (0.25).
# Uses 1d HTF for trend filter (more stable than 4h) and volume confirmation to reduce false breakouts.
# Camarilla pivot levels from 1d provide institutional structure; breakouts with volume confirm conviction.

name = "4h_Camarilla_R3S3_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed daily candle)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    camarilla_r3 = close_1d_vals + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d_vals - (high_1d - low_1d) * 1.1 / 2
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 34  # warmup for EMA34 and indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: price above/below 1d EMA34 determines trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 + uptrend + volume confirmation
            if curr_high > r3_aligned[i] and is_uptrend and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif curr_low < s3_aligned[i] and is_downtrend and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-30 14:41
