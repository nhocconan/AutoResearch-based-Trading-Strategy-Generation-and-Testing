# Strategy: mtf_12h_trix_camarilla_vol_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.444 | +9.7% | -8.1% | 188 | FAIL |
| ETHUSDT | -0.626 | +0.3% | -13.6% | 168 | FAIL |
| SOLUSDT | 0.426 | +43.0% | -20.4% | 161 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.306 | +23.0% | -8.7% | 65 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #024: 12h TRIX Trend + Camarilla Zone + Volume Spike

HYPOTHESIS: 12h timeframe captures significant institutional moves while
minimizing noise. TRIX(15) provides smooth trend confirmation without lag.
Camarilla S3/R3 levels are proven institutional support/resistance zones.
Volume spike confirms smart money participation. Choppiness filter avoids
whipsaws in ranging markets.

WHY 12h: Trade frequency target is 50-150 total (12-37/year). 4h strategies
overtraded (915 trades in current). 12h naturally limits trade count while
capturing multi-day moves. DONCHIAN(20) on 12h = 10-day breakout window.

WHY IT WORKS IN BULL + BEAR:
- Bull: TRIX>0 + price above Camarilla S3 + volume = trend continuation
- Bear: TRIX<0 + price below Camarilla R3 + volume = short rallies
- Range: Choppiness>61.8 → no trades (avoids whipsaws)

TARGET: 75-150 total trades over 4 years.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trix_camarilla_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_trix(close, period=15):
    """TRIX: triple smoothed EMA rate of change"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    trix = np.full(n, 0.0)
    for i in range(period * 3, n):
        if ema3[i - 1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]
    
    return trix

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values > 61.8 = choppy/range, < 38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period))
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
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
    
    # 1d EMA200 for multi-timeframe trend
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Local 12h indicators ===
    trix_15 = calculate_trix(close, period=15)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 200  # Need 200 for TRIX triple smoothing + EMA200
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix_15[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND: TRIX direction ===
        trix_bullish = trix_15[i] > 0
        trix_bearish = trix_15[i] < 0
        
        # === HTF TREND: Price vs 1d EMA200 ===
        above_htf_ema = close[i] > ema_200_aligned[i]
        below_htf_ema = close[i] < ema_200_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        chop = chop_14[i]
        in_chop = chop > 61.8 if not np.isnan(chop) else False
        in_trend_regime = chop < 50 if not np.isnan(chop) else True
        
        # === CAMARILLA LEVELS (previous bar) ===
        prev_close = close[i - 1]
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_range = prev_high - prev_low
        
        r1 = prev_close + prev_range * 0.09167
        r2 = prev_close + prev_range * 0.18333
        r3 = prev_close + prev_range * 0.275
        s1 = prev_close - prev_range * 0.09167
        s2 = prev_close - prev_range * 0.18333
        s3 = prev_close - prev_range * 0.275
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === DONCHIAN BREAKOUT ===
        # Use shift(1) to avoid look-ahead
        donchian_broken_up = close[i] > donchian_up[i - 1]
        donchian_broken_down = close[i] < donchian_lo[i - 1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Primary: TRIX bullish + above HTF EMA + Donchian breakout + volume
            # Secondary: TRIX bullish + touching S3 level + volume
            if trix_bullish and above_htf_ema:
                # Donchian breakout confirmation
                if donchian_broken_up and vol_spike:
                    desired_signal = SIZE
                # Camarilla S3 bounce
                elif vol_spike and low[i] <= s3:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Primary: TRIX bearish + below HTF EMA + Donchian breakdown + volume
            # Secondary: TRIX bearish + touching R3 level + volume
            if trix_bearish and below_htf_ema:
                # Donchian breakdown confirmation
                if donchian_broken_down and vol_spike:
                    desired_signal = -SIZE
                # Camarilla R3 rejection
                elif vol_spike and high[i] >= r3:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            bars_held = i - entry_bar
            
            if position_side > 0:
                # Long stop: lowest low since entry - 1 ATR buffer
                lowest_10 = np.min(low[entry_bar:i+1]) if i > entry_bar else low[entry_bar]
                stop_price = lowest_10 - 0.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                # Also exit if TRIX flips bearish
                if trix_bearish and bars_held >= 3:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stop: highest high since entry + 1 ATR buffer
                highest_10 = np.max(high[entry_bar:i+1]) if i > entry_bar else high[entry_bar]
                stop_price = highest_10 + 0.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                # Also exit if TRIX flips bullish
                if trix_bullish and bars_held >= 3:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 2:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-30 06:51
