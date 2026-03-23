# Strategy: mtf_12h_dual_hma_donchian_1d_1w_confirm_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.327 | +12.8% | -11.0% | 203 | FAIL |
| ETHUSDT | -0.844 | -2.3% | -11.7% | 207 | FAIL |
| SOLUSDT | 0.300 | +36.5% | -10.7% | 204 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.899 | +17.1% | -3.9% | 83 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #012: 12h Dual-HMA Trend + Donchian Breakout with 1d/1w Confirmation

Hypothesis: After 11 failures with regime-switching (Choppiness) strategies, return to 
proven trend-following with proper HTF confirmation. This combines:

1. 12H HMA(21) vs HMA(48) crossover - primary trend signal with less lag than EMA
2. 1D HMA(21) - confirms major trend direction (filter false breakouts)
3. 1W HMA(48) - secular trend bias (only long if weekly bullish)
4. Donchian(20) breakout - entry trigger on trend confirmation
5. ATR(14) trailing stop - 2.5 ATR exit to protect capital

Why this should work when others failed:
- Simpler logic = fewer conflicting filters = more trades generated
- 12h timeframe = 20-50 trades/year naturally (not too many fees, not too few signals)
- Triple HTF confirmation (1d + 1w) = filters 2022 crash whipsaws
- Donchian breakout = catches sustained moves, not noise
- Discrete sizing (0.25/0.30) = minimizes fee churn

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_hma_donchian_1d_1w_confirm_atr_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_hma_crossover(hma_fast, hma_slow):
    """Detect HMA crossover signals."""
    crossover_long = np.zeros(len(hma_fast))
    crossover_short = np.zeros(len(hma_fast))
    
    for i in range(1, len(hma_fast)):
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            continue
        if np.isnan(hma_fast[i-1]) or np.isnan(hma_slow[i-1]):
            continue
        
        # Long: fast crosses above slow
        if hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]:
            crossover_long[i] = 1
        
        # Short: fast crosses below slow
        if hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]:
            crossover_short[i] = 1
    
    return crossover_long, crossover_short

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Calculate 1W indicators
    hma_1w_48 = calculate_hma(df_1w['close'].values, 48)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_48_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_48)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_48 = calculate_hma(close, 48)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # HMA crossover signals on 12h
    crossover_long_12h, crossover_short_12h = calculate_hma_crossover(hma_12h_21, hma_12h_48)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_LONG = 0.28
    BASE_SIZE_SHORT = 0.25  # Slightly smaller for shorts (bear market bias)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_1w_48_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W SECULAR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_48_aligned[i]
        weekly_bearish = close[i] < hma_1w_48_aligned[i]
        
        # === 1D MAJOR TREND CONFIRMATION ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA TREND ===
        hma_bullish = hma_12h_21[i] > hma_12h_48[i]
        hma_bearish = hma_12h_21[i] < hma_12h_48[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === HMA CROSSOVER SIGNAL ===
        hma_cross_long = crossover_long_12h[i] == 1
        hma_cross_short = crossover_short_12h[i] == 1
        
        # === POSITION SIZING ===
        # Use discrete levels
        long_size = BASE_SIZE_LONG
        short_size = BASE_SIZE_SHORT
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Multiple confluence required
        # Need: 12h HMA bullish OR crossover + 1d bullish + breakout OR weekly bullish
        long_conditions = 0
        if hma_bullish:
            long_conditions += 1
        if hma_cross_long:
            long_conditions += 1
        if daily_bullish:
            long_conditions += 1
        if breakout_long:
            long_conditions += 1
        if weekly_bullish:
            long_conditions += 1
        
        # Enter long if 3+ conditions met (including at least one breakout/crossover)
        if long_conditions >= 3 and (breakout_long or hma_cross_long):
            new_signal = long_size
        
        # SHORT ENTRY: Multiple confluence required
        short_conditions = 0
        if hma_bearish:
            short_conditions += 1
        if hma_cross_short:
            short_conditions += 1
        if daily_bearish:
            short_conditions += 1
        if breakout_short:
            short_conditions += 1
        if weekly_bearish:
            short_conditions += 1
        
        # Enter short if 3+ conditions met (including at least one breakout/crossover)
        if short_conditions >= 3 and (breakout_short or hma_cross_short):
            new_signal = -short_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h HMA turns bearish strongly
            if position_side > 0 and hma_bearish and not hma_cross_long:
                trend_reversal = True
            # Exit short if 12h HMA turns bullish strongly
            if position_side < 0 and hma_bullish and not hma_cross_short:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 20:00
