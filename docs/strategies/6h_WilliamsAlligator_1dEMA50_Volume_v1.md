# Strategy: 6h_WilliamsAlligator_1dEMA50_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.177 | +28.3% | -11.0% | 244 | PASS |
| ETHUSDT | 0.206 | +30.8% | -17.9% | 230 | PASS |
| SOLUSDT | 0.651 | +88.1% | -22.6% | 192 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.617 | -0.2% | -6.4% | 74 | FAIL |
| ETHUSDT | 0.323 | +10.5% | -10.5% | 75 | PASS |
| SOLUSDT | -0.419 | -1.2% | -9.7% | 70 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 1d trend filter and volume confirmation.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 6h volume median.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 6h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Williams Alligator identifies trend phases via smoothed medians, reducing whipsaw in ranging markets.
# 1d EMA50 provides higher-timeframe trend filter. Volume confirmation avoids low-momentum breakouts.
# Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years) to minimize fee drag.
# Works in bull markets via trend-following entries and in bear markets via short alignments with volume.

name = "6h_WilliamsAlligator_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h Williams Alligator (Jaw=13, Teeth=8, Lips=5)
    # Jaw: Smoothed Median Price (13 periods)
    mp = (high + low) / 2
    jaw = pd.Series(mp).rolling(window=13, min_periods=13).median().values
    jaw = pd.Series(jaw).rolling(window=8, min_periods=8).mean().values  # Smoothed
    
    # Teeth: Smoothed Median Price (8 periods)
    teeth = pd.Series(mp).rolling(window=8, min_periods=8).median().values
    teeth = pd.Series(teeth).rolling(window=5, min_periods=5).mean().values  # Smoothed
    
    # Lips: Smoothed Median Price (5 periods)
    lips = pd.Series(mp).rolling(window=5, min_periods=5).median().values
    lips = pd.Series(lips).rolling(window=3, min_periods=3).mean().values  # Smoothed
    
    # Calculate 1d EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume median (20-period for stability)
    vol_median_6h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Alligator, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Williams Alligator alignment
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 6h volume median
        if vol_median_6h[i] <= 0 or np.isnan(vol_median_6h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_6h[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment AND uptrend AND volume spike
            if bullish_alignment and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish Alligator alignment AND downtrend AND volume spike
            elif bearish_alignment and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment turns bearish OR trend turns down
            elif not bullish_alignment or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment turns bullish OR trend turns up
            elif not bearish_alignment or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 10:18
