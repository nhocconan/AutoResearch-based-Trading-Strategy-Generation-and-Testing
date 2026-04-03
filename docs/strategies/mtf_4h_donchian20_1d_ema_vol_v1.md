# Strategy: mtf_4h_donchian20_1d_ema_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.199 | +29.7% | -17.5% | 173 | PASS |
| ETHUSDT | 0.400 | +46.4% | -10.2% | 160 | PASS |
| SOLUSDT | 0.488 | +67.3% | -30.4% | 162 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.774 | +19.6% | -11.1% | 57 | PASS |
| SOLUSDT | 1.461 | +35.0% | -7.4% | 50 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian(20) + 1d EMA + Volume Spike + ATR Stoploss

HYPOTHESIS: This replicates the proven winning pattern from DB:
- mtf_4h_donchian_vol_1w_ema_v1 had Sharpe 0.420 but only 26 trades
- mtf_4h_donchian16_vol_1d_ema_v1 had Sharpe 0.600 with 159 trades

Key insight: Use period=20 (not 16) for proper 5-day structure on 4h.
Add volume spike (>1.8x) as confirmation filter to reduce false breakouts.
1d EMA as trend filter (long only when price > EMA, short only when price < EMA).

WHY BOTH MARKETS:
- 2021 bull: Donchian breakout captures rallies, ATR protects during crashes
- 2022 bear: Trend filter prevents longs, shorts on breakdowns work
- 2025 range: EMA filter keeps us flat in chop, volume spike reduces whipsaws

TRADE COUNT: 75-200 total over 4 years (target 20-50/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) - 5 days on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # Default size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # 20 for Donchian + buffer
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === HTF TREND (1d EMA aligned) ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME SPIKE (>1.8x average) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT (close crosses outside prior bar's channel) ===
        # Use prior bar's channel for entry signal (no look-ahead)
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = False
        bearish_breakout = False
        
        if not np.isnan(prev_upper) and not np.isnan(prev_lower):
            # Close breaks above prior upper = bullish breakout
            bullish_breakout = close[i] > prev_upper
            # Close breaks below prior lower = bearish breakout
            bearish_breakout = close[i] < prev_lower
        
        # === MINIMUM HOLD: 4 bars (16h) to avoid immediate reversals ===
        min_hold_bars = 4
        min_hold = (i - entry_bar) >= min_hold_bars
        
        # === ATR TRAILING STOP (2.5x ATR from entry high/low) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                # Long stop: price fell below highest - 2.5*ATR
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                # Short stop: price rose above lowest + 2.5*ATR
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Also exit on trend reversal (trend filter flips)
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bullish breakout + volume spike + HTF bullish
            if bullish_breakout and vol_spike and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Bearish breakdown + volume spike + HTF bearish
            elif bearish_breakout and vol_spike and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-30 12:39
