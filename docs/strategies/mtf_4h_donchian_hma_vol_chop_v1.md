# Strategy: mtf_4h_donchian_hma_vol_chop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.087 | +17.9% | -10.4% | 337 | FAIL |
| ETHUSDT | 0.339 | +35.7% | -15.4% | 320 | PASS |
| SOLUSDT | 0.262 | +36.0% | -17.1% | 331 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.285 | +9.0% | -9.6% | 118 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #023: Donchian Breakout + HMA + Volume + Choppiness Regime (4h)

HYPOTHESIS: Use faster trend indicators (HMA16 instead of EMA200) to avoid
the "price never crosses EMA200" problem. Choppiness regime filters range markets.
HTF 12h EMA21 provides direction without being too restrictive.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Price breaks Donchian high + volume spike + above HMA16 = strong momentum
- Bear: Price breaks Donchian low + volume spike + below HMA16 = strong short
- Range (CHOP > 61.8): Skip entries, avoid whipsaws
- Trending (CHOP < 38.2): Allow entries in trend direction

EXPECTED TRADES: 100-200 total over 4 years (25-50/year)
- Donchian breaks every ~20-40 bars → ~219-438 potential/year
- Volume spike (1.5x) → reduces by ~40%
- HMA16 trend filter → reduces by ~30%
- Choppiness regime → reduces by ~20%
- Final: ~75-150 trades = statistical validity
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    n = len(data)
    half_length = period // 2
    full_length = period
    
    wma_half = pd.Series(data).rolling(window=half_length, min_periods=half_length).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    wma_full = pd.Series(data).rolling(window=full_length, min_periods=full_length).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    hma = 2 * wma_half - wma_full
    hma = pd.Series(hma).rolling(window=int(np.sqrt(period)), min_periods=1).mean().values
    return hma

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market choppiness vs trending
    CHOP > 61.8 = choppy (range-bound), CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            atr_sum = np.sum([
                max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                for j in range(i-period+1, i+1)
            ])
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # HTF EMA21 (faster, less restrictive than EMA200)
    ema21_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # === Local 4h indicators ===
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # HMA16 for local trend (faster than EMA200)
    hma16 = calculate_hma(close, 16)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 50  # Enough for HMA16, ATR14, Donchian20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma16[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === Regime check ===
        is_trending = chop[i] < 50.0  # Below 50 = trending (less strict than 38.2)
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Local trend (HMA16)
        local_bull = close[i] > hma16[i]
        local_bear = close[i] < hma16[i]
        
        # HTF trend (12h EMA21)
        htf_bull = ema21_12h_aligned[i] > ema21_12h_aligned[i-1] if not np.isnan(ema21_12h_aligned[i-1]) else False
        htf_bear = ema21_12h_aligned[i] < ema21_12h_aligned[i-1] if not np.isnan(ema21_12h_aligned[i-1]) else False
        
        # === LONG ENTRY: Price breaks above Donchian high + volume spike ===
        if not in_position:
            # Check for bullish breakout
            prev_donchian = donchian_upper[i-1] if i > 0 else np.nan
            bullish_breakout = high[i] > prev_donchian if not np.isnan(prev_donchian) else False
            
            # All conditions for long
            long_conditions = bullish_breakout and vol_spike and local_bull and is_trending
            
            if long_conditions:
                desired_signal = SIZE
                
            # === SHORT ENTRY: Price breaks below Donchian low + volume spike ===
            prev_donchian_low = donchian_lower[i-1] if i > 0 else np.nan
            bearish_breakout = low[i] < prev_donchian_low if not np.isnan(prev_donchian_low) else False
            
            # All conditions for short
            short_conditions = bearish_breakout and vol_spike and local_bear and is_trending
            
            if short_conditions:
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest point
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price falls below HMA16 (trend reversal)
                if close[i] < hma16[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest point
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price rises above HMA16 (trend reversal)
                if close[i] > hma16[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-30 11:21
