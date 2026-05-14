# Strategy: 4h_Donchian20_Breakout_1dHMA21_VolumeSpike_ATR_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.025 | +20.6% | -11.7% | 150 | PASS |
| ETHUSDT | 0.406 | +47.2% | -14.2% | 151 | PASS |
| SOLUSDT | 0.195 | +31.6% | -30.7% | 151 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.227 | -6.8% | -10.5% | 62 | FAIL |
| ETHUSDT | 0.102 | +6.9% | -11.0% | 58 | PASS |
| SOLUSDT | 0.061 | +6.1% | -11.8% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume spike confirmation.
# Long when price breaks above 20-period high AND 1d HMA21 uptrend AND volume > 2.0x 20-period median.
# Short when price breaks below 20-period low AND 1d HMA21 downtrend AND volume > 2.0x 20-period median.
# Uses ATR-based stoploss: exit long if price < highest_high_since_entry - 2.5*ATR(14),
# exit short if price > lowest_low_since_entry + 2.5*ATR(14).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year on 4h timeframe.
# HMA is smoother than EMA with less lag, improving trend reliability in both bull and bear markets.

name = "4h_Donchian20_Breakout_1dHMA21_VolumeSpike_ATR_v1"
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
    
    # Calculate 1d HMA21 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate HMA: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half = df_1d['close'].rolling(window=21//2, min_periods=21//2).mean()
    full = df_1d['close'].rolling(window=21, min_periods=21).mean()
    raw_hma = 2 * half - full
    hma_21_1d = raw_hma.rolling(window=int(np.sqrt(21)), min_periods=int(np.sqrt(21))).mean().values
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels (using lookback of 20 periods, excluding current)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for HMA, Donchian, volume, and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d HMA21 direction
        uptrend = curr_close > hma_21_1d_aligned[i]
        downtrend = curr_close < hma_21_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above 20-period high AND uptrend AND volume spike
            if curr_close > high_20[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Price breaks below 20-period low AND downtrend AND volume spike
            elif curr_close < low_20[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Donchian break OR trend reversal
            stop_price = highest_since_entry - 2.5 * curr_atr
            if curr_close < stop_price or curr_close < low_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Donchian break OR trend reversal
            stop_price = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stop_price or curr_close > high_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 13:07
