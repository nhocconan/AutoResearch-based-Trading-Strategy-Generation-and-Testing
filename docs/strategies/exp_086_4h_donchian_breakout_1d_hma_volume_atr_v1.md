# Strategy: exp_086_4h_donchian_breakout_1d_hma_volume_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.058 | +21.5% | -19.1% | 111 | PASS |
| ETHUSDT | -0.115 | +6.3% | -27.1% | 111 | FAIL |
| SOLUSDT | 1.135 | +257.9% | -25.5% | 96 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.291 | +11.1% | -19.7% | 32 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #086: 4h Donchian(20) Breakout + 1d HMA Trend Filter + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian breakouts on 4h aligned with 1d HMA(21) trend direction capture strong momentum with better trend alignment than 12h. 
Volume confirmation (>1.5x average) filters false breakouts. ATR-based stoploss (2.5x) manages risk. 
Uses 1d timeframe for trend filter to reduce noise and improve signal quality. Target: 75-200 trades over 4 years (19-50/year).
Works in both bull/bear by taking breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_086_4h_donchian_breakout_1d_hma_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Indicators: Donchian Channels (20-period) on 4h ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === HTF: 1d HMA(21) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values.astype(np.float64)
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def calculate_wma(data, period):
        if period <= 1:
            return data.copy()
        weights = np.arange(1, period + 1)
        return pd.Series(data).rolling(window=period, min_periods=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True).values
    
    def calculate_hma(data, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = calculate_wma(data, half_period)
        wma_full = calculate_wma(data, period)
        raw_hma = 2 * wma_half - wma_full
        hma = calculate_wma(raw_hma, sqrt_period)
        return hma
    
    hma_1d = calculate_hma(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)  # shift(1) applied
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Donchian, ATR, volume, HMA stability
    
    for i in range(warmup, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike: 1.5x average
        price_above_hma = price > hma_1d_aligned[i]
        price_below_hma = price < hma_1d_aligned[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_price = entry_price - 2.5 * entry_atr
                if price < stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_price = entry_price + 2.5 * entry_atr
                if price > stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Donchian opposite break exit (with volume confirmation)
            if position_side > 0:  # Long
                if price < donchian_lower[i-1] and vol_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > donchian_upper[i-1] and vol_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 1 bar to prevent whipsaw
            if bars_since_entry < 1:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long entry: price breaks above Donchian upper with volume AND price above 1d HMA (uptrend)
        if price > donchian_upper[i-1] and vol_spike and price_above_hma:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short entry: price breaks below Donchian lower with volume AND price below 1d HMA (downtrend)
        elif price < donchian_lower[i-1] and vol_spike and price_below_hma:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 13:12
