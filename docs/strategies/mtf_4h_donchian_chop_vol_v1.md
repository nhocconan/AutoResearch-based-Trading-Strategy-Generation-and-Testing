# Strategy: mtf_4h_donchian_chop_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.525 | +61.4% | -16.1% | 158 | PASS |
| ETHUSDT | 0.202 | +32.5% | -21.1% | 154 | PASS |
| SOLUSDT | 0.923 | +212.5% | -29.1% | 155 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.442 | -13.8% | -18.3% | 62 | FAIL |
| SOLUSDT | 0.344 | +12.8% | -13.1% | 49 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian + Choppiness Regime + Volume Spike

HYPOTHESIS: Choppiness Index is the proven meta-regime filter from DB winners.
- CHOP > 61.8 = choppy/range → NO entries (avoid false breakouts)
- CHOP < 50 = trending → allow entries

WHY IT SHOULD WORK IN BOTH MARKETS:
1. CHOP filters out choppy periods where breakouts fail (2022 crash was choppy at 4h)
2. Donchian(16) captures structural breaks (20 bars = ~3.3 days, reasonable frequency)
3. Volume spike confirms the breakout is institutional
4. 4h timeframe proven to avoid fee drag (vs 15m/30m)

ENTRY CONDITIONS (2 conditions + volume):
- CHOP < 50 (not choppy)
- Donchian(16) breakout (prior bar close breaks outside prior 15-bar range)
- Volume > 1.5x 20-bar MA

TRADE COUNT ESTIMATE:
- 4h bars/4yr ≈ 8760
- Donchian(16) breakout: ~1 per 30-50 bars = ~175-290 raw signals
- CHOP < 50 filter: ~50% of bars qualify = ~87-145
- Volume spike filter: ~40% pass = ~35-58 trades/symbol
- SAFE RANGE: 35-60 trades over 4 years per symbol

This is on the lower end but ACCEPTABLE if Sharpe > 0.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_vol_v1"
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

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - vectorized
    - Values > 61.8 indicate choppy/range market (avoid entries)
    - Values < 38.2 indicate strong trending
    - Values < 50 indicate not choppy (allow entries)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Precompute rolling high/low using numpy stride tricks or simple loop
    for i in range(period, n):
        period_high = high[i-period+1:i+1].max()
        period_low = low[i-period+1:i+1].min()
        
        if period_high > period_low:
            # Sum of ATR-like ranges
            sum_tr = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                sum_tr += tr
            
            if period_high != period_low:
                chop[i] = 100 * np.log10(sum_tr / (period_high - period_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Choppiness Index - regime filter
    chop = calculate_chop(high, low, close, period=14)
    
    # Donchian Channel(16) - price structure
    # Use 16 period to get breakout signals more frequently
    donchian_upper = pd.Series(high).rolling(window=16, min_periods=16).max().values
    donchian_lower = pd.Series(low).rolling(window=16, min_periods=16).min().values
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # Enough for Donchian16, ATR14, CHOP14
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: CHOP < 50 (not choppy) ===
        choppy_market = chop[i] > 50.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (prior bar's range) ===
        # Get prior bar's channel values
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior bar's upper channel
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        
        # Bearish breakout: close below prior bar's lower channel
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === MINIMUM HOLD: 2 bars ===
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_hit = low[i] < (entry_price - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (entry_price + 2.5 * entry_atr)
            
            # Exit on opposite breakout (trend reversal)
            reversal_exit = (position_side > 0 and bearish_breakout) or \
                           (position_side < 0 and bullish_breakout)
            
            if stop_hit:
                # Stopped out
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold and reversal_exit:
                # Trend reversal - exit
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                # Maintain position
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Skip if choppy market
            if choppy_market:
                signals[i] = 0.0
                continue
            
            # LONG: Bullish breakout + volume spike
            if bullish_breakout and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume spike
            elif bearish_breakout and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-30 12:16
