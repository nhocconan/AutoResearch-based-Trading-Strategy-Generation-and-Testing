# Strategy: mtf_12h_donchian_sma200_vol_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.386 | +11.9% | -8.0% | 62 | FAIL |
| ETHUSDT | -0.485 | +7.3% | -9.9% | 59 | FAIL |
| SOLUSDT | 0.588 | +55.0% | -13.1% | 57 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.441 | +10.7% | -5.2% | 20 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #002: 12h Donchian Breakout + 1d Trend + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 12h capture meaningful directional moves
while 1d SMA200 keeps us aligned with the broader trend. Volume confirmation
ensures institutional participation. ATR stoploss protects against false breakouts.

WHY IT WORKS IN BULL + BEAR:
- Bull: Price breaks Donchian upper + above SMA200 → long, rides trend
- Bear: Price breaks Donchian lower + below SMA200 → short, rides downtrend
- Volume filter avoids false breakouts (common in low-liquidity periods)
- 12h timeframe = fewer trades than 4h/6h, less fee drag, more meaningful moves

TARGET: 75-150 total trades over 4 years (19-37/year)
- Donchian(20) on 12h = ~1-2 breakouts per week naturally
- SMA200 filter cuts ~50% of signals (counter-trend)
- Volume >1.5x cuts another ~30%
- Expected: ~1 trade every 1-2 weeks = 75-130 over 4 years

Signal size: 0.28 (discrete, manageable drawdown in 2022 crash).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_sma200_vol_1d_v1"
timeframe = "12h"
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
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
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
    
    # 1d SMA200 for trend filter
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # === Local 12h indicators ===
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
    
    warmup = 250  # Need 200 for SMA200 + 20 for Donchian + 20 for volume MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND: Price vs 1d SMA200 ===
        above_htf_sma = close[i] > sma_200_aligned[i]
        below_htf_sma = close[i] < sma_200_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # 50% above average (strict filter)
        
        # === DONCHIAN BREAKOUT ===
        # Previous bar's Donchian levels (breakout confirmation)
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        # Breakout signals
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Donchian breakout up + above SMA200 + volume spike ===
            if breakout_up and above_htf_sma and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Donchian breakout down + below SMA200 + volume spike ===
            elif breakout_down and below_htf_sma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR from entry) ===
        if in_position:
            if position_side > 0:
                # Update highest high since entry for trailing stop
                if i == entry_bar or high[i] > highest_since_entry:
                    highest_since_entry = high[i]
                
                # Trailing stop: highest high - 2.5 ATR
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if trend reverses (price crosses below SMA200)
                if below_htf_sma:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update lowest low since entry for trailing stop
                if i == entry_bar or low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                
                # Trailing stop: lowest low + 2.5 ATR
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if trend reverses (price crosses above SMA200)
                if above_htf_sma:
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
2026-03-30 07:23
