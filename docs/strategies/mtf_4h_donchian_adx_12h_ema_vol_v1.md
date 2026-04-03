# Strategy: mtf_4h_donchian_adx_12h_ema_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.372 | +46.7% | -24.4% | 126 | PASS |
| ETHUSDT | 0.248 | +37.3% | -19.3% | 133 | PASS |
| SOLUSDT | 1.136 | +304.6% | -25.0% | 123 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.426 | -12.9% | -15.8% | 53 | FAIL |
| SOLUSDT | 0.427 | +15.3% | -17.1% | 41 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #023: 4h Donchian + ADX Trend Strength + 12h EMA Direction + Volume

HYPOTHESIS: Combine proven DB winner elements with ADX for better trend strength
quantification vs CHOP alone.

CORE ELEMENTS (from DB winners):
1. Donchian(20) breakout - proven structural break detection
2. ADX > 20 - quantifies trend strength (vs CHOP which only detects range)
3. 12h EMA direction - HTF trend filter prevents countertrend trades
4. Volume spike - institutional confirmation

WHY IT SHOULD WORK IN BOTH MARKETS:
- ADX>20 confirms directional momentum exists (not just CHOP<50)
- 12h EMA filter ensures we're trading WITH HTF trend, not against it
- 2022 crash was choppy with ADX spikes - this catches those directional moves
- 2025 bear is range-bound with occasional breaks - ADX filters false breakouts

TRADE COUNT ESTIMATE:
- ADX>20: ~40-50% of bars
- 12h EMA aligned: ~60% of ADX signals = ~24-30 signals
- Donchian breakout: ~60-70% pass rate = ~15-20 signals
- Volume spike: ~70% pass rate = ~10-15 trades/symbol/year
- 4yr total: ~40-60 trades - slightly low but CHOP<50 optional relax

Adding CHOP<50 as secondary filter to get into 60-80 range.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_adx_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """
    ADX (Average Directional Index) - vectorized approximation
    Measures trend strength, NOT direction.
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method (EWM with alpha=1/period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if atr[i] > 1e-10:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
    
    # Calculate DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX as smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    Choppiness Index (CHOP) - secondary regime filter
    CHOP > 61.8 = choppy, CHOP < 50 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        period_high = high[i-period+1:i+1].max()
        period_low = low[i-period+1:i+1].min()
        
        if period_high > period_low:
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
    
    # === HTF: 12h EMA for trend direction (call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    chop_14 = calculate_chop(high, low, close, period=14)
    
    # Donchian Channel(20) for breakout structure
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
    
    warmup = 60  # Need enough for ADX, EMA12h alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTERS ===
        # ADX > 20: trend has strength
        trend_strong = adx_14[i] > 20.0
        
        # CHOP < 55: not choppy (relaxed from 50 to allow more trades)
        not_choppy = chop_14[i] < 55.0 if not np.isnan(chop_14[i]) else True
        
        # === HTF TREND DIRECTION (12h EMA aligned) ===
        htf_bullish = close[i] > ema_12h_aligned[i]
        htf_bearish = close[i] < ema_12h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === DONCHIAN BREAKOUT (prior bar's range) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior bar's upper channel
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        
        # Bearish breakout: close below prior bar's lower channel
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === MINIMUM HOLD: 3 bars ===
        min_hold = (i - entry_bar) >= 3
        
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
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold and reversal_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Need regime + HTF alignment + breakout + volume
            if not (trend_strong and not_choppy):
                signals[i] = 0.0
                continue
            
            # LONG: HTF bullish + bullish breakout + volume spike
            if htf_bullish and bullish_breakout and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: HTF bearish + bearish breakout + volume spike
            elif htf_bearish and bearish_breakout and vol_spike:
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
2026-03-30 12:27
