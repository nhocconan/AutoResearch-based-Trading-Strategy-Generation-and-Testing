# Strategy: mtf_1d_donchian_hma_volume_1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.122 | +0.2% | -7.7% | 43 | FAIL |
| ETHUSDT | -1.431 | -5.7% | -7.7% | 36 | FAIL |
| SOLUSDT | 0.213 | +28.9% | -13.7% | 32 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.066 | +6.5% | -4.0% | 14 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #107: 1d Primary + 1w HTF — Donchian Breakout with Volume Confirmation

Hypothesis: Previous dual-regime strategies (#103, #106) failed because regime detection
adds complexity and lag. Pure trend-following with Donchian breakouts has proven success
on 4h (current best Sharpe=0.486). This adapts it to 1d with:

1) 1w HMA(21) for macro trend bias — only trade breakouts in trend direction
2) Donchian(20) breakout — price breaks 20-day high/low for entry
3) Volume confirmation — breakout volume > 1.5x 20-day avg (filters false breakouts)
4) ATR(14) trailing stop at 2.5x — locks in profits, limits drawdown
5) Simple exit: opposite Donchian break OR trend reversal

Why this should work on 1d:
- Donchian breakouts are proven trend-following signals (Turtle Trading)
- 1w HMA filter prevents counter-trend trades in bear markets (2022 crash)
- Volume filter reduces whipsaws in ranging markets
- 1d naturally produces 20-40 trades/year (low fee drag)
- Simpler logic = more robust across BTC/ETH/SOL

Position size: 0.25 base, 0.30 max with volume confluence
Stoploss: 2.5*ATR trailing
Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_volume_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-day high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1w HMA slope (trend strength)
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]) and hma_1w_aligned[i-1] != 0:
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1] * 100
        else:
            hma_1w_slope[i] = 0.0
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        hma_slope_positive = hma_1w_slope[i] > 0.5
        hma_slope_negative = hma_1w_slope[i] < -0.5
        hma_slope_flat = abs(hma_1w_slope[i]) <= 0.5
        
        # === 1d TREND FILTER ===
        hma_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === DONCHIAN BREAKOUT ===
        prev_high = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_low = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        breakout_long = close[i] > prev_high
        breakout_short = close[i] < prev_low
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.5
        volume_strong = volume_ratio > 2.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 1w trend up OR flat + 1d trend up + Donchian breakout + volume
        if price_above_hma_1w or hma_slope_flat:
            if hma_1d_bullish and breakout_long:
                if volume_confirmed:
                    new_signal = POSITION_SIZE_BASE
                    if volume_strong and hma_slope_positive:
                        new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Require: 1w trend down OR flat + 1d trend down + Donchian breakout + volume
        if price_below_hma_1w or hma_slope_flat:
            if hma_1d_bearish and breakout_short:
                if volume_confirmed:
                    new_signal = -POSITION_SIZE_BASE
                    if volume_strong and hma_slope_negative:
                        new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold long if still above Donchian mid and 1w trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if close[i] > donchian_mid[i] and price_above_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if close[i] < donchian_mid[i] and price_below_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_1w and hma_slope_negative:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if breakout_short:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1w and hma_slope_positive:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if breakout_long:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        # Simple RSI calculation for take profit
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)
        avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_14 = 100.0 - (100.0 / (1.0 + rs))
        
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 04:56
