# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ATR_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.369 | +41.2% | -10.9% | 182 | PASS |
| ETHUSDT | 0.342 | +42.9% | -12.5% | 176 | PASS |
| SOLUSDT | 0.722 | +113.0% | -25.7% | 166 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.576 | -1.0% | -8.4% | 77 | FAIL |
| ETHUSDT | 0.773 | +20.6% | -9.9% | 63 | PASS |
| SOLUSDT | 0.512 | +15.2% | -10.4% | 58 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level (from 1d) AND 1d EMA34 uptrend AND volume > 2.0x 20-period median.
# Short when price breaks below Camarilla S3 level (from 1d) AND 1d EMA34 downtrend AND volume > 2.0x 20-period median.
# Uses ATR(14) stoploss: exit long if price < highest_since_entry - 2.5*ATR(14), exit short if price > lowest_since_entry + 2.5*ATR(14).
# Discrete position sizing (0.25) minimizes fee churn. Target: 19-50 trades/year on 4h timeframe.
# Camarilla levels from 1d provide strong support/resistance; volume confirms breakout validity; 1d EMA34 filters counter-trend noise.
# This combination targets the 4h timeframe with HTF=1d to reduce noise and improve Sharpe in both bull and bear markets.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ATR_v1"
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
    
    # Calculate 1d EMA34 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla R3 = close + 1.1 * (high - low) / 2
    # Camarilla S3 = close - 1.1 * (high - low) / 2
    # Using previous 1d bar's OHLC to avoid look-ahead
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    camarilla_r3_1d = df_1d_close + 1.1 * (df_1d_high - df_1d_low) / 2
    camarilla_s3_1d = df_1d_close - 1.1 * (df_1d_high - df_1d_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for EMA34, ATR, and volume median
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i-1]  # Using previous bar's R3 level
        breakout_down = curr_low < camarilla_s3_aligned[i-1]  # Using previous bar's S3 level
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume spike
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Breakout down AND downtrend AND volume spike
            elif breakout_down and downtrend and volume_confirm:
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
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = highest_since_entry - 2.5 * curr_atr
            if curr_close < stop_price or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stop_price or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 14:03
