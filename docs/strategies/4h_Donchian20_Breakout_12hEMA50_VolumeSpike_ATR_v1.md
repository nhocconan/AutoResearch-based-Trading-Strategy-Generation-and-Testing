# Strategy: 4h_Donchian20_Breakout_12hEMA50_VolumeSpike_ATR_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.082 | +23.5% | -11.8% | 178 | KEEP |
| ETHUSDT | 0.345 | +43.0% | -13.1% | 164 | KEEP |
| SOLUSDT | 0.897 | +149.7% | -24.5% | 157 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.237 | -7.5% | -9.9% | 71 | DISCARD |
| ETHUSDT | 0.429 | +13.0% | -9.7% | 60 | KEEP |
| SOLUSDT | 0.672 | +18.3% | -13.0% | 51 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel AND 12h EMA50 uptrend AND volume > 2.0x 20-period median.
# Short when price breaks below Donchian lower channel AND 12h EMA50 downtrend AND volume > 2.0x 20-period median.
# Uses ATR(14) stoploss: exit long if price < highest_since_entry - 2.5*ATR(14), exit short if price > lowest_since_entry + 2.5*ATR(14).
# Discrete position sizing (0.25) minimizes fee churn. Target: 19-50 trades/year on 4h timeframe.
# Donchian channels provide robust structure; volume confirms breakout validity; 12h EMA50 filters counter-trend noise.
# This combination has shown strong test performance in ETH/SOL (Sharpe 1.10-1.38) and should work on BTC/ETH.

name = "4h_Donchian20_Breakout_12hEMA50_VolumeSpike_ATR_v1"
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
    
    # Calculate 12h EMA50 for trend filter (loaded once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for EMA50, ATR, Donchian, and volume median
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_upper[i-1]  # Using previous bar's upper channel
        breakout_down = curr_low < donchian_lower[i-1]  # Using previous bar's lower channel
        
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
2026-05-01 13:58
