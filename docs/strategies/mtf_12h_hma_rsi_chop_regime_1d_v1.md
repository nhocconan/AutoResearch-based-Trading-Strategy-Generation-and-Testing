# Strategy: mtf_12h_hma_rsi_chop_regime_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.240 | -33.0% | -35.1% | 185 | FAIL |
| ETHUSDT | -0.619 | -17.6% | -29.5% | 172 | FAIL |
| SOLUSDT | 0.054 | +17.6% | -26.4% | 177 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.010 | +27.2% | -11.2% | 66 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #086: 12h Primary + 1d HTF — Simplified Trend Pullback with Choppiness Regime

Hypothesis: Previous 12h strategies failed due to overly complex regime switching causing 0 trades.
This version uses SIMPLER entry conditions while keeping the proven 12h/1d MTF structure.

Key changes from failures:
1) LOOSEN RSI entry thresholds (RSI<50 for long, RSI>50 for short) - previous was too strict
2) Choppiness Index is OPTIONAL filter, not required - allows trades in all regimes
3) 1d HMA slope is primary trend filter (proven in #079)
4) 12h HMA crossover for entry timing (simpler than CRSI/Donchian)
5) Discrete sizing: 0.28 base, 0.35 max with confluence
6) ATR(14) trailing stoploss at 2.5x

Why this should work:
- Simpler conditions = more trades across ALL symbols (BTC/ETH/SOL)
- 12h timeframe naturally limits trades to 20-50/year
- 1d HMA prevents counter-trend trades in bear markets (2025 test)
- Choppiness adds edge but doesn't block all trades
- Proven MTF structure from #079 (which generated trades)

Position size: 0.28 base, 0.35 max with confluence
Stoploss: 2.5*ATR trailing
Target: 25-45 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_chop_regime_1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope (trend strength)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.28
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(ema_21[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_slope_positive = hma_1d_slope[i] > 0.1  # slight positive slope
        hma_slope_negative = hma_1d_slope[i] < -0.1  # slight negative slope
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 55.0  # trending market (looser than 38.2)
        chop_ranging = chop_14[i] > 45.0  # ranging market (looser than 61.8)
        
        # === 12h HMA CROSSOVER ===
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # === RSI ENTRY SIGNALS (LOOSE thresholds for trade generation) ===
        rsi_neutral_long = rsi_14[i] < 55.0  # not overbought
        rsi_neutral_short = rsi_14[i] > 45.0  # not oversold
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === EMA CONFIRMATION ===
        ema_bullish = close[i] > ema_21[i]
        ema_bearish = close[i] < ema_21[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d uptrend + 12h bullish + RSI okay ---
        # Primary: 1d HMA bullish + 12h HMA crossover + RSI not overbought
        if price_above_hma_1d and hma_bullish and rsi_neutral_long:
            new_signal = POSITION_SIZE_BASE
            # Boost if trending regime + EMA confirmation
            if chop_trending and ema_bullish:
                new_signal = POSITION_SIZE_MAX
            # Boost if RSI oversold (pullback entry)
            elif rsi_oversold:
                new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: 1d downtrend + 12h bearish + RSI okay ---
        # Primary: 1d HMA bearish + 12h HMA crossover + RSI not oversold
        if price_below_hma_1d and hma_bearish and rsi_neutral_short:
            new_signal = -POSITION_SIZE_BASE
            # Boost if trending regime + EMA confirmation
            if chop_trending and ema_bearish:
                new_signal = -POSITION_SIZE_MAX
            # Boost if RSI overbought (pullback entry)
            elif rsi_overbought:
                new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Keep position if RSI hasn't reached extreme exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 75.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 25.0:
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
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 1d HMA turns bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_slope_negative:
                new_signal = 0.0
        
        # Exit short if 1d HMA turns bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
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
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 04:36
