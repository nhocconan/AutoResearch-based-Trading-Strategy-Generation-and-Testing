# Strategy: mtf_4h_chop_donchian_vol_regime_12h_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.185 | +26.9% | -6.4% | 298 | PASS |
| ETHUSDT | -0.016 | +20.0% | -6.1% | 308 | FAIL |
| SOLUSDT | 0.363 | +43.6% | -18.9% | 312 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.491 | +23.9% | -6.2% | 107 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #003: 4h Choppiness Regime + Donchian Breakout + Volume

HYPOTHESIS: Choppiness Index is the KEY missing ingredient. Most strategies
fail because they enter during ranging markets (CHOP > 61.8). By only entering
when CHOP < 50 (trending), we dramatically improve win rate.

WHY IT WORKS IN BULL + BEAR + RANGE:
- Bull: CHOP < 50 + Donchian breakout up + HTF trend up = high win rate longs
- Bear: CHOP < 50 + Donchian breakout down + HTF trend down = high win rate shorts
- Range: CHOP > 61.8 = NO ENTRIES (avoids the #1 killer: whipsaws in chop)
- ATR stoploss handles volatility scaling automatically

KEY INSIGHT FROM DB:
- mtf_4h_crsi_chop_donchian_regime_1d_v1: test Sharpe 1.46 (392 trades)
- gen_camarilla_pivot_volume_spike_choppiness_4h_v1: test Sharpe 1.47 (95 trades)
Both use Choppiness Index as the regime filter - this is NOT optional.

TARGET: 100-250 total trades over 4 years (25-62/year)
- CHOP < 50 filter cuts ~60% of potential signals (ranging = skip)
- Donchian breakout = ~1-2/week naturally
- Volume > 1.5x = ~30% additional filter
- Expected: ~1 trade/week = 200 over 4 years

Signal size: 0.28 (discrete, survives 2022 crash with <30% drawdown).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_donchian_vol_regime_12h_v1"
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
    """Donchian Channel breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (choppy) - DON'T enter
    CHOP < 50 = trending - GOOD to enter
    
    Formula: 100 * LOG10(SUM(ATR(1), period) / ( HighestHigh - LowestLow )) / LOG10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR(1) for each bar
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(21) for trend direction
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    
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
    
    warmup = 250  # Need 200 for Donchian + 14 for CHOP + 20 for volume MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        # THIS IS THE KEY: skip entries during choppy markets
        # CHOP > 61.8 = ranging = DON'T enter
        # CHOP < 50 = trending = OK to enter
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 50
        
        # === HTF TREND: 12h EMA(21) direction ===
        htf_trend_up = close[i] > ema_aligned[i]
        htf_trend_down = close[i] < ema_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Trending + breakout up + HTF trend up + volume spike ===
            if breakout_up and htf_trend_up and vol_spike and is_trending:
                desired_signal = SIZE
            # === LONG: Very strong trend (no volume needed) ===
            elif breakout_up and htf_trend_up and is_trending and chop_value < 40:
                desired_signal = SIZE * 0.8  # Smaller size without volume confirm
            
            # === SHORT: Trending + breakout down + HTF trend down + volume spike ===
            if breakout_down and htf_trend_down and vol_spike and is_trending:
                desired_signal = -SIZE
            # === SHORT: Very strong trend (no volume needed) ===
            elif breakout_down and htf_trend_down and is_trending and chop_value < 40:
                desired_signal = -SIZE * 0.8
        
        # === STOPLOSS (2.5 ATR from entry) ===
        if in_position:
            if position_side > 0:
                # Update highest high since entry
                if i == entry_bar or high[i] > highest_since_entry:
                    highest_since_entry = high[i]
                
                # Trailing stop
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update lowest low since entry
                if i == entry_bar or low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                
                # Trailing stop
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
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
2026-03-30 07:25
