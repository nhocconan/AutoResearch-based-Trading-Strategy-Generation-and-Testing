# Strategy: 6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.176 | +29.4% | -14.5% | 95 | PASS |
| ETHUSDT | 0.184 | +30.6% | -18.5% | 89 | PASS |
| SOLUSDT | 0.868 | +159.3% | -32.3% | 86 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.963 | -6.5% | -11.5% | 45 | FAIL |
| ETHUSDT | 0.571 | +17.5% | -8.9% | 31 | PASS |
| SOLUSDT | -0.143 | +0.9% | -18.8% | 29 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 6h timeframe with 12h EMA50 trend filter and volume spike confirmation (>2.0x average volume).
Only trade breakouts aligned with 12h EMA50 direction during high volume expansion. Uses discrete sizing (0.25) and ATR-based stoploss.
Designed to capture strong momentum moves in both bull and bear markets by using the 12h EMA as a dynamic trend filter.
Targets 12-37 trades/year on 6h timeframe to minimize fee drag.
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
    
    # Get 6h data for Camarilla levels and ATR - primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate previous period's Camarilla levels (using prior 6h bar's HLC)
    # We need to shift the high/low/close by 1 to get the previous bar's values
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    # Set first value to NaN since there's no previous bar
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for R3, R4, S3, S4
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    rang = prev_high - prev_low
    r4 = prev_close + (rang * 1.1 / 2)
    r3 = prev_close + (rang * 1.1 / 4)
    s3 = prev_close - (rang * 1.1 / 4)
    s4 = prev_close - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4)
    
    # Get 12h data for EMA50 trend filter - HTF
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate ATR(14) for stoploss on 6h
    # True Range
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_6h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14)  # EMA needs 50, vol needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or 
            np.isnan(atr_6h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        r4_val = r4_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        s4_val = s4_aligned[i]
        ema_val = ema_12h_aligned[i]
        atr_val = atr_6h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend and volume confirmation
            # Long: price breaks above R3, above 12h EMA50, with volume spike
            long_signal = (high_val > r3_val) and (close_val > ema_val) and volume_spike
            # Short: price breaks below S3, below 12h EMA50, with volume spike
            short_signal = (low_val < s3_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes below 12h EMA50
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes above 12h EMA50
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 23:07
