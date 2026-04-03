# Strategy: mtf_12h_camarilla_vol_1w_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.492 | +4.0% | -13.0% | 8 | FAIL |
| ETHUSDT | -0.784 | -17.8% | -31.2% | 7 | FAIL |
| SOLUSDT | 2.360 | +881.9% | -22.1% | 353 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.531 | +40.1% | -9.3% | 116 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #008: 12h Camarilla Pivot + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Camarilla pivot levels on 12h timeframe provide high-probability reversal/breakout zones. 
Combined with 1d volume spike (>2.0x average) and 1week trend filter (price > EMA50 for longs, < EMA50 for shorts), 
this strategy captures institutional interest at key levels. Target: 12-37 trades/year on 12h (50-150 total over 4 years) 
to minimize fee drag. Uses discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d_pivot = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d bar
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_p = np.full(n, np.nan)
    
    if len(df_1d_pivot) >= 2:
        high_1d = df_1d_pivot['high'].values
        low_1d = df_1d_pivot['low'].values
        close_1d = df_1d_pivot['close'].values
        
        # Calculate pivot and levels for each 1d bar
        p_1d = (high_1d + low_1d + close_1d) / 3.0
        range_1d = high_1d - low_1d
        
        h4_1d = p_1d + range_1d * 1.1 / 2
        l4_1d = p_1d - range_1d * 1.1 / 2
        h3_1d = p_1d + range_1d * 1.1 / 4
        l3_1d = p_1d - range_1d * 1.1 / 4
        h2_1d = p_1d + range_1d * 1.1 / 6
        l2_1d = p_1d - range_1d * 1.1 / 6
        h1_1d = p_1d + range_1d * 1.1 / 12
        l1_1d = p_1d - range_1d * 1.1 / 12
        
        # Align to 12h timeframe (shift by 1 to avoid look-ahead)
        camarilla_p_aligned = align_htf_to_ltf(prices, df_1d_pivot, p_1d)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d_pivot, h4_1d)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d_pivot, l4_1d)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d_pivot, h3_1d)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d_pivot, l3_1d)
        camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d_pivot, h2_1d)
        camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d_pivot, l2_1d)
        camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d_pivot, h1_1d)
        camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d_pivot, l1_1d)
    else:
        camarilla_p_aligned = np.full(n, np.nan)
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_h2_aligned = np.full(n, np.nan)
        camarilla_l2_aligned = np.full(n, np.nan)
        camarilla_h1_aligned = np.full(n, np.nan)
        camarilla_l1_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 1week EMA50 ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Camarilla H3 with volume confirmation and uptrend
        long_condition = (
            close[i] > camarilla_h3_aligned[i] and 
            volume_spike and 
            price_above_1w_ema
        )
        
        # Short: Price breaks below Camarilla L3 with volume confirmation and downtrend
        short_condition = (
            close[i] < camarilla_l3_aligned[i] and 
            volume_spike and 
            price_below_1w_ema
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 10:21
