# Strategy: exp_354_1h_camarilla_4h_1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.566 | -22.2% | -38.7% | 86 | FAIL |
| SOLUSDT | 0.919 | +177.2% | -26.3% | 100 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.087 | +6.2% | -18.3% | 35 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #354: 1h Camarilla Pivot + 4h/1d Trend + Volume Spike

HYPOTHESIS: Camarilla pivot levels from 1d provide key support/resistance zones. 
Breakouts above R4 or below S4 with volume confirmation (>1.5x average) and aligned 
trend (4h close > 4h EMA50 AND 1d close > 1d EMA50) capture strong momentum moves. 
Use 4h/1d for signal direction, 1h only for entry timing to minimize fee drag. 
Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_354_1h_camarilla_4h_1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours (08-20 UTC) to avoid per-bar datetime conversion
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # === HTF: 4h data for trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === HTF: 1d data for Camarilla pivots and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    def calculate_camarilla(h, l, c):
        """Calculate Camarilla pivot levels: R4, R3, R2, R1, PP, S1, S2, S3, S4"""
        range_ = h - l
        pp = (h + l + c) / 3.0
        r4 = c + range_ * 1.1 / 2.0
        r3 = c + range_ * 1.1 / 4.0
        r2 = c + range_ * 1.1 / 6.0
        r1 = c + range_ * 1.1 / 12.0
        s1 = c - range_ * 1.1 / 12.0
        s2 = c - range_ * 1.1 / 6.0
        s3 = c - range_ * 1.1 / 4.0
        s4 = c - range_ * 1.1 / 2.0
        return r4, r3, r2, r1, pp, s1, s2, s3, s4
    
    # Calculate for each 1d bar
    r4_1d = np.full(len(df_1d), np.nan)
    r3_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    s4_1d = np.full(len(df_1d), np.nan)
    pp_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = calculate_camarilla(
            df_1d['high'].iloc[i], 
            df_1d['low'].iloc[i], 
            df_1d['close'].iloc[i]
        )
        r4_1d[i] = r4
        r3_1d[i] = r3
        s3_1d[i] = s3
        s4_1d[i] = s4
        pp_1d[i] = pp
    
    # Align Camarilla levels to 1h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr_1h = np.zeros(n)
    tr_1h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_1h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital) - reduced to lower drawdown
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for 1d indicators stability
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Require alignment across 4h and 1d ---
        price = close[i]
        trend_4h_up = price > ema50_4h_aligned[i]
        trend_1d_up = price > ema50_1d_aligned[i]
        is_uptrend = trend_4h_up and trend_1d_up
        trend_4h_down = price < ema50_4h_aligned[i]
        trend_1d_down = price < ema50_1d_aligned[i]
        is_downtrend = trend_4h_down and trend_1d_down
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price Levels ---
        r4 = r4_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        pp = pp_1d_aligned[i]
        
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
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: Price > R4 + volume spike + uptrend alignment
        long_breakout = (price > r4) and volume_spike and is_uptrend
        
        # Short breakout: Price < S4 + volume spike + downtrend alignment
        short_breakout = (price < s4) and volume_spike and is_downtrend
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout:
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
2026-04-03 12:30
