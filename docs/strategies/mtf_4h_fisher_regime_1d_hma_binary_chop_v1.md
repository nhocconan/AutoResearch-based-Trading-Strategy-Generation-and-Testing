# Strategy: mtf_4h_fisher_regime_1d_hma_binary_chop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.107 | +11.0% | -20.1% | 431 | FAIL |
| ETHUSDT | 0.386 | +51.0% | -18.2% | 429 | PASS |
| SOLUSDT | 0.404 | +64.2% | -41.1% | 439 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.128 | +2.0% | -13.9% | 135 | FAIL |
| SOLUSDT | 0.032 | +4.7% | -19.2% | 138 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #351: 4h Primary + 1d HTF — Fisher Transform Reversals with Regime Filter

Hypothesis: Previous 4h strategies failed because:
1. Too many confluence filters prevented trades from triggering (0 trades = auto-reject)
2. Choppiness thresholds too narrow (45/55) - rarely triggered regime switches
3. RSI extremes (30/70) too common in strong trends, causing whipsaws

This strategy uses Ehlers Fisher Transform for cleaner reversal signals:
1. 1d HMA(21) as MACRO BIAS (only long if price > 1d HMA, only short if price < 1d HMA)
2. 4h Choppiness Index for regime (CHOP>50=range, CHOP<50=trend) - SIMPLER binary split
3. RANGE REGIME: Fisher Transform extremes (<-1.5 long, >+1.5 short) + RSI confirmation
4. TREND REGIME: HMA(16/48) crossover + Fisher confirms direction
5. ATR(14) trailing stop at 2.5x for risk management
6. RELAXED thresholds to ensure 25-50 trades/year on 4h

KEY INSIGHT: Fisher Transform normalizes price into Gaussian distribution, making
extremes (-2 to +2) more reliable than RSI for reversals. Combined with 1d HMA bias,
this should work in both bull and bear markets.

TARGET: 25-50 trades/year on 4h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_regime_1d_hma_binary_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price into a Gaussian normal distribution for cleaner reversal signals.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    # Calculate typical price
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    # Normalize price over lookback period
    highest = typical_s.rolling(window=period, min_periods=period).max().values
    lowest = typical_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize to 0-1 range
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (typical - lowest) / (highest - lowest + 1e-10)
    
    # Clamp to avoid division issues
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Apply Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    
    # Signal line (1-period lag)
    fisher_s = pd.Series(fisher)
    fisher_prev = fisher_s.shift(1).values
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # HMA for trend detection (fast and slow)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    
    # Fisher Transform for reversals
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
    # Calculate and align 1d HMA for macro bias (HARD FILTER)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 4h (target 25-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA - HARD FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index - BINARY) ===
        is_choppy = chop[i] > 50.0  # High choppiness = range regime (mean revert)
        is_trending = chop[i] <= 50.0  # Low choppiness = trend regime (breakout)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Fisher Transform mean reversion
            # Long: Fisher < -1.5 + crosses up + price above 1d HMA
            # Short: Fisher > +1.5 + crosses down + price below 1d HMA
            
            fisher_oversold = fisher[i] < -1.5
            fisher_overbought = fisher[i] > 1.5
            fisher_cross_up = fisher[i] > fisher_prev[i] and fisher_prev[i] < -1.0
            fisher_cross_down = fisher[i] < fisher_prev[i] and fisher_prev[i] > 1.0
            
            if price_above_hma_1d and fisher_oversold and fisher_cross_up:
                # Long oversold in bullish macro (range regime)
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and fisher_overbought and fisher_cross_down:
                # Short overbought in bearish macro (range regime)
                desired_signal = -BASE_SIZE
            
            # Fallback: RSI extremes if Fisher doesn't trigger
            elif price_above_hma_1d and rsi_14[i] < 28:
                desired_signal = BASE_SIZE * 0.7
            
            elif price_below_hma_1d and rsi_14[i] > 72:
                desired_signal = -BASE_SIZE * 0.7
        
        elif is_trending:
            # TREND REGIME: HMA crossover + Fisher confirms direction
            # Long: HMA16 > HMA48 + Fisher > -1.0 + 1d bullish
            # Short: HMA16 < HMA48 + Fisher < +1.0 + 1d bearish
            
            hma_bullish = hma_16[i] > hma_48[i]
            hma_bearish = hma_16[i] < hma_48[i]
            fisher_bullish = fisher[i] > -1.0
            fisher_bearish = fisher[i] < 1.0
            
            if price_above_hma_1d and hma_bullish and fisher_bullish:
                # Long trend in bullish macro (trend regime)
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and hma_bearish and fisher_bearish:
                # Short trend in bearish macro (trend regime)
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === FISHER EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and fisher[i] > 1.5:
            # Long position: exit when Fisher reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            # Short position: exit when Fisher reaches oversold
            desired_signal = 0.0
        
        # === RSI EXIT (extreme reached) ===
        if in_position and position_side > 0 and rsi_14[i] > 70:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if regime and bias still valid
            if position_side > 0:
                if price_above_hma_1d:
                    if (is_choppy and fisher[i] < 1.5 and rsi_14[i] < 70) or \
                       (is_trending and hma_16[i] > hma_48[i]):
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                if price_below_hma_1d:
                    if (is_choppy and fisher[i] > -1.5 and rsi_14[i] > 30) or \
                       (is_trending and hma_16[i] < hma_48[i]):
                        desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 09:01
