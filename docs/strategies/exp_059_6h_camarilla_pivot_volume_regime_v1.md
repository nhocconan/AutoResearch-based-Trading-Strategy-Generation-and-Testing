# Strategy: exp_059_6h_camarilla_pivot_volume_regime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.066 | +22.7% | -13.1% | 179 | PASS |
| ETHUSDT | 0.527 | +58.2% | -14.1% | 164 | PASS |
| SOLUSDT | 0.431 | +62.8% | -32.6% | 141 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.133 | +7.4% | -8.2% | 58 | PASS |
| SOLUSDT | 0.130 | +7.3% | -16.4% | 49 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #059: 6h Camarilla Pivot + Volume Spike + Regime Filter (ADX)
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from 12h timeframe act as strong support/resistance.
Breakouts above R4 or below S4 with volume confirmation (>1.8x average) and ADX>25 indicate strong momentum.
In ranging markets (ADX<20), fade at R3/S3 levels. This adapts to both bull/bear regimes by using ADX
to determine market state and applying appropriate logic. Target: 75-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_059_6h_camarilla_pivot_volume_regime_v1"
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
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous day) ===
    def calculate_camarilla(high, low, close):
        # Typical price for pivot
        pivot = (high + low + close) / 3.0
        range_ = high - low
        # Camarilla levels
        r4 = pivot + (range_ * 1.1 / 2)
        r3 = pivot + (range_ * 1.1 / 4)
        s3 = pivot - (range_ * 1.1 / 4)
        s4 = pivot - (range_ * 1.1 / 2)
        return r3, r4, s3, s4, pivot
    
    # Calculate for each 12h bar (using previous bar's data)
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    camarilla_r3 = np.full_like(c_12h, np.nan)
    camarilla_r4 = np.full_like(c_12h, np.nan)
    camarilla_s3 = np.full_like(c_12h, np.nan)
    camarilla_s4 = np.full_like(c_12h, np.nan)
    camarilla_pivot = np.full_like(c_12h, np.nan)
    
    for i in range(1, len(c_12h)):
        r3, r4, s3, s4, p = calculate_camarilla(h_12h[i-1], l_12h[i-1], c_12h[i-1])
        camarilla_r3[i] = r3
        camarilla_r4[i] = r4
        camarilla_s3[i] = s3
        camarilla_s4[i] = s4
        camarilla_pivot[i] = p
    
    # Align to 6h timeframe
    camarilla_r3_6h = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    camarilla_pivot_6h = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    
    # === 6h Indicators: ADX(14) for regime detection ===
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
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
    bars_since_entry = 0
    
    warmup = 50  # Warmup for ADX and volume stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx[i]) or np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_r4_6h[i]) or
            np.isnan(camarilla_s3_6h[i]) or np.isnan(camarilla_s4_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.8  # Volume spike threshold
        
        # --- Regime Detection ---
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions based on regime
            if is_trending:
                # In trending market: exit on opposite Camarilla break with volume
                if position_side > 0:  # Long
                    if low[i] < camarilla_s3_6h[i] and vol_spike:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if high[i] > camarilla_r3_6h[i] and vol_spike:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Ranging market
                # Exit when price returns to pivot or opposite level
                if position_side > 0:  # Long
                    if price < camarilla_pivot_6h[i]:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if price > camarilla_pivot_6h[i]:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            
            # Minimum holding period of 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if is_trending:
            # Trending market: breakout continuation at R4/S4 with volume
            if price > camarilla_r4_6h[i-1] and vol_spike:  # Break above R4
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < camarilla_s4_6h[i-1] and vol_spike:  # Break below S4
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif is_ranging:
            # Ranging market: fade at R3/S3 levels
            if price < camarilla_r3_6h[i-1] and price > camarilla_pivot_6h[i-1] and vol_spike:
                # Near R3 from below - short fade
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif price > camarilla_s3_6h[i-1] and price < camarilla_pivot_6h[i-1] and vol_spike:
                # Near S3 from above - long fade
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            # Transition regime (ADX between 20-25) - no trade
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 13:03
