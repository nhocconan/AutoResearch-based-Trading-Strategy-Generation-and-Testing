# Strategy: mtf_4h_donchian_ema200_vol_spike_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.576 | +9.3% | -5.3% | 189 | FAIL |
| ETHUSDT | -0.693 | +4.0% | -9.3% | 177 | FAIL |
| SOLUSDT | 0.766 | +61.4% | -8.0% | 189 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.314 | +9.0% | -3.8% | 67 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian Breakout + 1d EMA200 Trend + Volume Spike

HYPOTHESIS: Donchian(20) breakouts on 4h capture institutional moves while
1d EMA200 filters ensure we only trade with the higher timeframe trend.
Volume spike (>1.5x 20-bar MA) confirms smart money participation.
This pattern appears in multiple DB winners (SOL Sharpe 1.10-1.38, ETH Sharpe 1.47).

WHY IT WORKS IN BULL + BEAR:
- Bull: Price > 1d EMA200 + Donchian breakout up = trend continuation
- Bear: Price < 1d EMA200 + Donchian breakdown = short rallies
- Volume filter avoids false breakouts (major cause of whipsaws)

WHY 4h: Trade frequency target 75-200 total over 4 years. 4h Donchian(20)
= 80-hour breakout window, captures multi-day moves without overtrading.

TARGET: 100-200 total trades over 4 years (25-50/year).
Signal size: 0.28 (discrete, manageable 27% max drawdown on 77% crash).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_ema200_vol_spike_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - use shift(1) to avoid look-ahead"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for multi-timeframe trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 220  # Need 200 for EMA200 + 20 for Donchian
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND: Price vs 1d EMA200 ===
        above_htf_ema = close[i] > ema_200_aligned[i]
        below_htf_ema = close[i] < ema_200_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # 50% above average
        
        # === DONCHIAN BREAKOUT (use shift(1) to avoid look-ahead) ===
        donchian_broken_up = close[i] > donchian_up[i - 1]
        donchian_broken_down = close[i] < donchian_lo[i - 1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Above 1d EMA200 + Donchian breakout + volume ===
            if above_htf_ema and donchian_broken_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT ENTRY: Below 1d EMA200 + Donchian breakdown + volume ===
            if below_htf_ema and donchian_broken_down and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            if position_side > 0:
                # Update highest high since entry for trailing stop
                if i == entry_bar or close[i] > highest_since_entry:
                    highest_since_entry = high[i]
                
                # Trailing stop: highest high - 2.5 ATR
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update lowest low since entry for trailing stop
                if i == entry_bar or low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                
                # Trailing stop: lowest low + 2.5 ATR
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-30 07:17
