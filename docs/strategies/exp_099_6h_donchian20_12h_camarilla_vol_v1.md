# Strategy: exp_099_6h_donchian20_12h_camarilla_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.094 | +24.0% | -10.5% | 198 | PASS |
| ETHUSDT | 0.709 | +84.9% | -18.0% | 174 | PASS |
| SOLUSDT | 1.156 | +249.4% | -27.7% | 152 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.036 | +3.9% | -10.5% | 68 | FAIL |
| SOLUSDT | 0.185 | +8.5% | -18.3% | 53 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #099: 6h Donchian(20) breakout + 12h Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for continuation) capture high-probability moves. Volume confirmation (>1.5x average) filters weak breakouts. Discrete position sizing (0.25) and ATR stoploss (2.0x) manage risk. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_099_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels for 12h
    def calculate_camarilla(high, low, close):
        # Typical price for pivot
        pt = (high + low + close) / 3.0
        rng = high - low
        # Camarilla levels
        r4 = pt + rng * 1.1 / 2
        r3 = pt + rng * 1.1 / 4
        r2 = pt + rng * 1.1 / 6
        r1 = pt + rng * 1.1 / 12
        s1 = pt - rng * 1.1 / 12
        s2 = pt - rng * 1.1 / 6
        s3 = pt - rng * 1.1 / 4
        s4 = pt - rng * 1.1 / 2
        return r3, r4, s3, s4  # We only need R3,R4,S3,S4
    
    r3_12h = np.full(len(df_12h), np.nan)
    r4_12h = np.full(len(df_12h), np.nan)
    s3_12h = np.full(len(df_12h), np.nan)
    s4_12h = np.full(len(df_12h), np.nan)
    
    for i in range(len(df_12h)):
        if i >= 0:  # Need at least one bar
            r3, r4, s3, s4 = calculate_camarilla(
                df_12h['high'].values[i],
                df_12h['low'].values[i],
                df_12h['close'].values[i]
            )
            r3_12h[i] = r3
            r4_12h[i] = r4
            s3_12h[i] = s3
            s4_12h[i] = s4
    
    # Align to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 6h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 60  # Warmup for Donchian channels and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_lower[i-1]  # Break below lower channel
        
        # --- Camarilla Pivot Conditions ---
        # R3/S3: Mean reversion zone (fade)
        # R4/S4: Breakout zone (continuation)
        near_r3 = abs(price - r3_12h_aligned[i]) / price < 0.005  # Within 0.5%
        near_s3 = abs(price - s3_12h_aligned[i]) / price < 0.005
        break_r4 = price > r4_12h_aligned[i]  # Break above R4
        break_s4 = price < s4_12h_aligned[i]  # Break below S4
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2R or if breaking S4 (for longs)
                if break_s4 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2R or if breaking R4 (for shorts)
                if break_r4 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # 1. Donchian breakout up AND volume spike AND (near R3 for mean reversion OR break above R4 for continuation)
        # 2. OR break above R4 with volume (strong continuation)
        if (breakout_up and volume_spike and (near_r3 or break_r4)) or (break_r4 and volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short conditions:
        # 1. Donchian breakout down AND volume spike AND (near S3 for mean reversion OR break below S4 for continuation)
        # 2. OR break below S4 with volume (strong continuation)
        elif (breakout_down and volume_spike and (near_s3 or break_s4)) or (break_s4 and volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 13:19
