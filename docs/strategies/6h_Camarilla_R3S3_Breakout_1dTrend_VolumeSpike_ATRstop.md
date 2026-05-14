# Strategy: 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_ATRstop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.249 | +29.4% | -6.7% | 269 | PASS |
| ETHUSDT | 0.549 | +43.7% | -7.4% | 215 | PASS |
| SOLUSDT | 0.571 | +63.6% | -13.0% | 191 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.915 | -1.3% | -5.7% | 126 | FAIL |
| ETHUSDT | 1.291 | +22.5% | -6.0% | 81 | PASS |
| SOLUSDT | 0.245 | +8.9% | -9.9% | 81 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_ATRstop
Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 in 1d uptrend (close > 1d EMA34) with volume > 2.0x 20-bar average.
Short when price breaks below Camarilla S3 in 1d downtrend (close < 1d EMA34) with volume > 2.0x 20-bar average.
Exit via ATR-based trailing stop (2.5*ATR from extreme) or re-entry into Camarilla H3/L3 range.
Designed for ~12-37 trades/year by requiring strong breakouts (R3/S3), daily trend alignment, and volume confirmation.
Works in bull/bear markets via 1d EMA34 filter; avoids whipsaws via volume confirmation and tight stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get previous day's OHLC for Camarilla levels (1d HTF, aligned with 1 extra delay for completed bar)
    prev_close = align_htf_to_ltf(prices, df_1d, df_1d['close'].values, additional_delay_bars=1)
    prev_high = align_htf_to_ltf(prices, df_1d, df_1d['high'].values, additional_delay_bars=1)
    prev_low = align_htf_to_ltf(prices, df_1d, df_1d['low'].values, additional_delay_bars=1)
    prev_open = align_htf_to_ltf(prices, df_1d, df_1d['open'].values, additional_delay_bars=1)
    
    # Camarilla levels from previous day's OHLC
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.1 / 12
    S1 = prev_close - camarilla_range * 1.1 / 12
    R3 = prev_close + camarilla_range * 1.1 / 4
    S3 = prev_close - camarilla_range * 1.1 / 4
    H3 = prev_close + camarilla_range * 1.1 / 2
    L3 = prev_close - camarilla_range * 1.1 / 2
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # highest close since long entry
    short_low = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA34 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: break above Camarilla R3 with volume spike
                long_signal = (close[i] > R3[i]) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: break below Camarilla S3 with volume spike
                short_signal = (close[i] < S3[i]) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_high = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_low = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest close
            if close[i] > long_high:
                long_high = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Camarilla H3/L3 range
            atr_stop = long_high - 2.5 * atr[i]
            range_exit = (close[i] < H3[i] and close[i] > L3[i])
            if close[i] <= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest close
            if close[i] < short_low:
                short_low = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Camarilla H3/L3 range
            atr_stop = short_low + 2.5 * atr[i]
            range_exit = (close[i] > L3[i] and close[i] < H3[i])
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_ATRstop"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 21:39
