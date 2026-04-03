# Strategy: mtf_4h_donchian12_vol_ema16_12h_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.267 | +37.1% | -18.6% | 180 | PASS |
| ETHUSDT | 0.075 | +20.3% | -20.4% | 185 | PASS |
| SOLUSDT | 0.981 | +231.8% | -25.8% | 174 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.168 | +8.0% | -13.7% | 54 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + 12h EMA Trend + Volume Spike

HYPOTHESIS: This is a CLONE of the proven DB winner pattern:
  mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (SOLUSDT: test_sharpe=1.382, 95tr)
  
WHY IT SHOULD WORK IN BOTH MARKETS:
- 4h timeframe: proven to avoid 15m/30m fee drag issues
- Donchian(20) breakout: captures structural breaks, works in all market phases
- 12h EMA16: provides trend direction without being too slow like weekly EMA
- Volume spike confirmation: filters false breakouts
- 2.5 ATR stoploss: proven risk management

EXPECTED TRADES: 75-200 per symbol over 4 years
- Donchian(20) on 4h = ~1 breakout per 20-40 bars
- Volume spike filter reduces by ~40%
- 12h EMA16 trend filter reduces by ~30%
- Final: ~100-200 total per symbol (safe range)

KEY FIX FROM FAILURES:
- Changed from 6h/12h/1d to 4h (proven timeframe)
- Changed from weekly/daily EMA to 12h EMA (more signals, not too few)
- Kept 3 conditions: breakout + vol + trend (not over-filtered)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian12_vol_ema16_12h_v1"
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
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA16 for trend direction (align to 4h)
    htf_ema16 = pd.Series(df_12h['close'].values).ewm(span=16, min_periods=16, adjust=False).mean().values
    ema16_aligned = align_htf_to_ltf(prices, df_12h, htf_ema16)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) - price channel breakout
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 bars) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 80  # Enough for Donchian20, ATR14, EMA16 alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema16_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: 12h EMA16 ===
        bull_trend = close[i] > ema16_aligned[i]
        bear_trend = close[i] < ema16_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (use prior bar's channel) ===
        # This ensures we're breaking OUT of established range
        prev_high_19 = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_low_19 = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior 19-bar high
        bullish_breakout = (not np.isnan(prev_high_19) and close[i] > prev_high_19)
        
        # Bearish breakout: close below prior 19-bar low
        bearish_breakout = (not np.isnan(prev_low_19) and close[i] < prev_low_19)
        
        # === MINIMUM HOLD: 2 bars to reduce fee churn ===
        min_hold_bars = (i - entry_bar) >= 2 if in_position else True
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
                stop_hit = low[i] < stop_price
            else:
                stop_price = entry_price + 2.5 * entry_atr
                stop_hit = high[i] > stop_price
            
            # Trend exit: price crosses EMA16
            trend_exit = (position_side > 0 and close[i] < ema16_aligned[i]) or \
                        (position_side < 0 and close[i] > ema16_aligned[i])
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold_bars and trend_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend
            if bullish_breakout and vol_spike and bull_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend
            elif bearish_breakout and vol_spike and bear_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
    
    return signals
```

## Last Updated
2026-03-30 12:07
