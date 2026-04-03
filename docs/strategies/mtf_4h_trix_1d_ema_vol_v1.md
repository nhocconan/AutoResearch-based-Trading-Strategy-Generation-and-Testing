# Strategy: mtf_4h_trix_1d_ema_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.082 | +23.6% | -17.2% | 247 | PASS |
| ETHUSDT | 0.450 | +51.3% | -13.3% | 238 | PASS |
| SOLUSDT | 0.785 | +123.8% | -26.6% | 243 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.953 | +22.9% | -8.0% | 72 | PASS |
| SOLUSDT | 0.326 | +11.3% | -10.8% | 83 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #024: 4h TRIX Crossover + 1d EMA Trend + Volume Confirmation

HYPOTHESIS: TRIX (triple smoothed momentum) catches trend changes earlier than
ADX while filtering noise better than RSI. The DB shows TRIX-based strategy
achieved test Sharpe=1.32 on ETH. Combined with 1d EMA for structure and
volume confirmation, this should capture major trends while avoiding chop.

WHY IT SHOULD WORK IN BOTH MARKETS:
- TRIX crossover is fast enough to catch 2022 crash momentum shifts
- 1d EMA ensures we're not fighting the larger trend
- Volume confirmation filters false breakouts
- 2025 bear/range has occasional directional moves - TRIX catches those
- TRIX's triple smoothing reduces whipsaws vs single EMA crossover

TRADE COUNT ESTIMATE:
- TRIX crossing signal line: ~15-20 crossover events/symbol/year
- 1d EMA alignment filter: ~60% pass = ~9-12 signals
- Volume spike (>1.3x): ~70% pass = ~6-8 trades/symbol/year
- 4yr total: ~24-32 per symbol - slightly low

ADDING Donchian(20) breakout as secondary confirmation to boost trades to 40-60.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=14):
    """
    TRIX (Triple EMA) - triple smoothed momentum oscillator
    TRIX = rate of change of triple EMA
    Values near 0 = no trend, crossing above/below 0 = momentum shift
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Triple EMA smoothing
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = percentage change of triple EMA
    trix = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if ema3[i-1] > 1e-10:
            trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
    
    # Signal line = EMA of TRIX
    signal = pd.Series(trix).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    return trix, signal

def calculate_rsi(prices, period=14):
    """RSI - Relative Strength Index"""
    n = len(prices)
    deltas = np.zeros(n, dtype=np.float64)
    deltas[1:] = prices[1:] - prices[:-1]
    
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    
    # === HTF: 1d EMA for trend direction (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix, signal_line = calculate_trix(close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian Channel for structure
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 80  # Need enough for TRIX triple smoothing
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(trix[i]) or np.isnan(signal_line[i]):
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
        
        # === TRIX CROSSOVER SIGNALS ===
        # Bullish: TRIX crosses above signal line (momentum shifting up)
        trix_cross_up = (trix[i-1] <= signal_line[i-1]) and (trix[i] > signal_line[i])
        
        # Bearish: TRIX crosses below signal line (momentum shifting down)
        trix_cross_down = (trix[i-1] >= signal_line[i-1]) and (trix[i] < signal_line[i])
        
        # === HTF TREND DIRECTION (1d EMA aligned) ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === DONCHIAN BREAKOUT (prior bar) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === MINIMUM HOLD: 4 bars ===
        min_hold = (i - entry_bar) >= 4
        
        # === EXITS ===
        if in_position:
            # ATR trailing stop
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * entry_atr)
            
            # Exit on TRIX reversal
            trix_reversal = (position_side > 0 and trix_cross_down) or \
                           (position_side < 0 and trix_cross_up)
            
            # Exit on RSI extreme
            rsi_extreme = (rsi_14[i] < 30.0) if position_side > 0 else (rsi_14[i] > 70.0)
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold and trix_reversal:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold and rsi_extreme:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Need HTF alignment + momentum signal + volume
            
            # LONG: HTF bullish + TRIX cross up + volume spike
            if htf_bullish and trix_cross_up and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: HTF bearish + TRIX cross down + volume spike
            elif htf_bearish and trix_cross_down and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # ALT ENTRY: HTF aligned + Donchian breakout + volume (catches more moves)
            elif htf_bullish and bullish_breakout and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            elif htf_bearish and bearish_breakout and vol_spike:
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
2026-03-30 12:34
