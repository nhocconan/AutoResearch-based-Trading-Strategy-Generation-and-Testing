# Strategy: 6h_1dCamarilla_R3S3_R4S4_BreakoutFade_1dEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.567 | +52.7% | -10.5% | 138 | PASS |
| ETHUSDT | 0.340 | +41.6% | -14.7% | 136 | PASS |
| SOLUSDT | 0.630 | +88.9% | -22.5% | 105 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.425 | +0.9% | -7.2% | 52 | FAIL |
| ETHUSDT | 0.985 | +23.5% | -8.7% | 45 | PASS |
| SOLUSDT | 0.038 | +5.7% | -10.5% | 45 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout)
# with 1d EMA34 trend filter and volume confirmation (>2.0x average). 
# Long when price breaks above R4 in uptrend (close > EMA34) with volume spike.
# Short when price breaks below S4 in downtrend (close < EMA34) with volume spike.
# Fade longs at R3 in downtrend (close < EMA34) and fade shorts at S3 in uptrend (close > EMA34).
# Uses ATR-based trailing stop (2.5x ATR) to manage risk.
# Designed for low trade frequency (~12-37/year on 6h) to minimize fee drag while capturing strong directional moves and mean reversion at key levels.
# Works in bull markets via R4 breakout continuation and in bear markets via S4 breakdown continuation.
# Camarilla levels from 1d provide institutional support/resistance that price respects.

name = "6h_1dCamarilla_R3S3_R4S4_BreakoutFade_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: based on previous day's range
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    r4 = close_1d + rng * 1.1 / 2.0
    r3 = close_1d + rng * 1.1 / 4.0
    s3 = close_1d - rng * 1.1 / 4.0
    s4 = close_1d - rng * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 6h timeframe (wait for 1d bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for dynamic trailing stop on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    start_idx = 50  # warmup for EMA(34) and Camarilla calculation
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma_20 = np.mean(volume[:i])
        else:
            vol_ma_20 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_r4 = r4_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_ema = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries or fades
            if volume_spike:
                # Breakout longs: price breaks above R4 in uptrend
                if curr_close > curr_r4 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Breakout shorts: price breaks below S4 in downtrend
                elif curr_close < curr_s4 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
                # Fade longs: price rejects at S3 in uptrend (mean reversion up)
                elif curr_close < curr_s3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Fade shorts: price rejects at R3 in downtrend (mean reversion down)
                elif curr_close > curr_r3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.5 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.5 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-30 12:33
