# Strategy: 4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_ATRstop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.089 | +24.1% | -12.1% | 607 | PASS |
| ETHUSDT | 0.115 | +25.4% | -20.4% | 517 | PASS |
| SOLUSDT | 0.395 | +49.5% | -29.3% | 359 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.936 | -2.5% | -7.7% | 255 | FAIL |
| ETHUSDT | 1.365 | +24.4% | -7.5% | 178 | PASS |
| SOLUSDT | 0.772 | +17.0% | -9.1% | 168 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_ATRstop
Hypothesis: 4h Camarilla R1/S1 breakout with 12h trend filter and volume spike confirmation.
Long when price breaks above Camarilla R1 in 12h uptrend (close > 12h EMA50) with volume > 2.0x 20-bar average.
Short when price breaks below Camarilla S1 in 12h downtrend (close < 12h EMA50) with volume > 2.0x 20-bar average.
Exit via ATR-based trailing stop (2.0*ATR from extreme) or re-entry into Camarilla H3/L3 range.
Designed for ~19-50 trades/year by requiring strong breakouts (R1/S1), 12h trend alignment, and volume confirmation.
Works in bull/bear markets via 12h EMA50 filter; avoids whipsaws via volume confirmation and ATR trailing stop.
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
    
    # Get 12h data for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get previous day's OHLC for Camarilla levels (1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
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
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_12h_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (12h EMA50 filter)
            if close[i] > ema_trend:  # 12h uptrend regime
                # Long: break above Camarilla R1 with volume spike
                long_signal = (close[i] > R1[i]) and vol_regime[i]
            else:  # 12h downtrend regime
                # Short: break below Camarilla S1 with volume spike
                short_signal = (close[i] < S1[i]) and vol_regime[i]
            
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
            atr_stop = long_high - 2.0 * atr[i]
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
            atr_stop = short_low + 2.0 * atr[i]
            range_exit = (close[i] > L3[i] and close[i] < H3[i])
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_ATRstop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 21:40
