# Strategy: 6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Session

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.072 | +23.3% | -5.9% | 260 | PASS |
| ETHUSDT | 0.138 | +25.2% | -7.2% | 224 | PASS |
| SOLUSDT | 0.270 | +33.2% | -16.4% | 188 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.845 | -3.2% | -3.6% | 105 | FAIL |
| ETHUSDT | 0.401 | +9.6% | -3.9% | 88 | PASS |
| SOLUSDT | 0.592 | +10.7% | -3.1% | 71 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level AND 1d close > 1d EMA34 (uptrend) AND 6h volume > 2.0x 20-period volume MA.
# Short when price breaks below Camarilla S3 level AND 1d close < 1d EMA34 (downtrend) AND 6h volume > 2.0x 20-period volume MA.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size fixed at 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla levels provide objective intraday support/resistance, 1d EMA34 filters for trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (OHLC)
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/2
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 6h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 2.0)
        
        # Camarilla breakout conditions
        breakout_up = high_val > camarilla_r3_aligned[i]  # Price breaks above R3
        breakout_down = low_val < camarilla_s3_aligned[i]  # Price breaks below S3
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Camarilla breakout up AND 1d uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down AND 1d downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla pivot point (PP) OR trend changes
            # Camarilla PP = (prev_high + prev_low + prev_close) / 3
            camarilla_pp = (prev_high + prev_low + prev_close) / 3
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
            if close_val < camarilla_pp_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla pivot point (PP) OR trend changes
            camarilla_pp = (prev_high + prev_low + prev_close) / 3
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
            if close_val > camarilla_pp_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 10:43
