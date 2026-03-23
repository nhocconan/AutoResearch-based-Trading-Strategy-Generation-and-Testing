# Strategy: mtf_4h_hma_cross_rsi_atr_1d_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.079 | -7.6% | -10.2% | 261 | FAIL |
| ETHUSDT | -1.231 | +1.2% | -7.2% | 241 | FAIL |
| SOLUSDT | 0.164 | +26.7% | -9.4% | 251 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.353 | +8.0% | -2.8% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #521: 4h Primary + 1d HTF — HMA Crossover + RSI Filter + ATR Trail

Hypothesis: After 467 failed strategies (mostly complex volspike/Fisher/Choppiness combos),
try a SIMPLER approach with fewer conflicting filters to ensure trade frequency.

Key insights from failures:
- Complex multi-condition entries (volspike + fisher + BB + RSI + Donchian) = 0 trades
- Too many filters = mutually exclusive conditions
- Simpler logic = more trades while maintaining quality

This strategy uses:
1. HMA crossover (16/48) on 4h for entry timing - faster response than EMA
2. 1d HMA(21) for major trend direction - only trade with HTF trend
3. RSI(14) filter to avoid overbought/oversold entries
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete position sizing (0.30) to minimize fee churn

Why this might work:
- HMA is proven to reduce lag vs EMA (research note #1)
- 1d trend filter prevents counter-trend trades (major failure mode)
- RSI filter avoids chasing extremes
- Simple logic = consistent signals across BTC/ETH/SOL
- 4h TF targets 25-40 trades/year (optimal fee/trade ratio)

Position sizing: 0.30 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_cross_rsi_atr_1d_v2"
timeframe = "4h"
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
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # HMA crossover signals (16/48)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    
    # RSI for entry filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track HMA crossover
    prev_hma_16 = np.zeros(n)
    prev_hma_48 = np.zeros(n)
    prev_hma_16[1:] = hma_4h_16[:-1]
    prev_hma_48[1:] = hma_4h_48[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength confirmation
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H HMA CROSSOVER SIGNALS ===
        # Bullish crossover: fast HMA crosses above slow HMA
        hma_cross_up = (hma_4h_16[i] > hma_4h_48[i]) and (prev_hma_16[i] <= prev_hma_48[i])
        # Bearish crossover: fast HMA crosses below slow HMA
        hma_cross_down = (hma_4h_16[i] < hma_4h_48[i]) and (prev_hma_16[i] >= prev_hma_48[i])
        
        # HMA alignment (already in trend)
        hma_aligned_bull = hma_4h_16[i] > hma_4h_48[i]
        hma_aligned_bear = hma_4h_16[i] < hma_4h_48[i]
        
        # === RSI FILTER (avoid extreme entries) ===
        rsi_neutral_long = rsi_14[i] < 70.0  # Not overbought for long
        rsi_neutral_short = rsi_14[i] > 30.0  # Not oversold for short
        rsi_oversold = rsi_14[i] < 35.0  # Good for long entry
        rsi_overbought = rsi_14[i] > 65.0  # Good for short entry
        
        # === ENTRY LOGIC — SIMPLE WITH CONFLUENCE ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions for frequency)
        # Condition 1: HMA crossover up + bull regime + RSI not overbought
        if hma_cross_up and bull_regime and rsi_neutral_long:
            new_signal = POSITION_SIZE
        # Condition 2: HMA aligned bull + bull regime + RSI oversold (pullback entry)
        elif hma_aligned_bull and bull_regime and rsi_oversold:
            new_signal = POSITION_SIZE
        # Condition 3: Strong bull (1d HMA slope) + HMA crossover
        elif bull_regime and hma_slope_bull and hma_cross_up:
            new_signal = POSITION_SIZE
        # Condition 4: HMA aligned + RSI oversold (mean reversion in uptrend)
        elif hma_aligned_bull and rsi_oversold and bull_regime:
            new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: HMA crossover down + bear regime + RSI not oversold
            if hma_cross_down and bear_regime and rsi_neutral_short:
                new_signal = -POSITION_SIZE
            # Condition 2: HMA aligned bear + bear regime + RSI overbought (bounce entry)
            elif hma_aligned_bear and bear_regime and rsi_overbought:
                new_signal = -POSITION_SIZE
            # Condition 3: Strong bear (1d HMA slope) + HMA crossover
            elif bear_regime and hma_slope_bear and hma_cross_down:
                new_signal = -POSITION_SIZE
            # Condition 4: HMA aligned + RSI overbought (mean reversion in downtrend)
            elif hma_aligned_bear and rsi_overbought and bear_regime:
                new_signal = -POSITION_SIZE * 0.8
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme RSI) ===
        # Exit long on regime flip to bear or extreme overbought
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
            elif rsi_14[i] > 80.0:  # Extreme overbought
                new_signal = 0.0
        
        # Exit short on regime flip to bull or extreme oversold
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
            elif rsi_14[i] < 20.0:  # Extreme oversold
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
                # Flip position
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
2026-03-23 05:29
