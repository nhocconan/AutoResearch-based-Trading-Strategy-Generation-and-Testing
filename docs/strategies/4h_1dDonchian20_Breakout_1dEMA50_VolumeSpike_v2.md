# Strategy: 4h_1dDonchian20_Breakout_1dEMA50_VolumeSpike_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.255 | +31.3% | -7.7% | 68 | PASS |
| ETHUSDT | 0.078 | +23.3% | -12.1% | 60 | PASS |
| SOLUSDT | 0.699 | +95.6% | -21.9% | 68 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.204 | -3.0% | -7.0% | 26 | FAIL |
| ETHUSDT | 1.225 | +24.7% | -6.2% | 25 | PASS |
| SOLUSDT | -0.213 | +2.3% | -9.5% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d HTF for Donchian channel calculation (upper/lower 20-period) for strong breakout signals and 1d EMA50 for trend filter.
# Long when price breaks above 1d Donchian upper in uptrend (4h close > 1d EMA50) with volume spike (>2.0x average).
# Short when price breaks below 1d Donchian lower in downtrend (4h close < 1d EMA50) with volume spike.
# Uses ATR-based trailing stop to manage risk and reduce whipsaw.
# Designed for low trade frequency (~19-50/year on 4h) to minimize fee drag while capturing strong directional moves.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at 1d Donchian levels.
# Focus on BTC/ETH as primary targets.

name = "4h_1dDonchian20_Breakout_1dEMA50_VolumeSpike_v2"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper: max(high, 20), lower: min(low, 20)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 4h timeframe (wait for 1d bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(14) for dynamic trailing stop on 4h
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
    
    start_idx = 50  # warmup for EMA(50) and Donchian(20)
    
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
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        curr_ema = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d Donchian upper with 1d uptrend (close > EMA50)
                if curr_close > curr_upper and curr_close > curr_ema:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Bearish entry: price breaks below 1d Donchian lower with 1d downtrend (close < EMA50)
                elif curr_close < curr_lower and curr_close < curr_ema:
                    signals[i] = -0.30
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
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.5 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-04-30 12:32
