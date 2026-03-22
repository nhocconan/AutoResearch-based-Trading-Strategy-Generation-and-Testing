#!/usr/bin/env python3
"""
Experiment #244: 4h Fisher Transform Reversal Strategy with 1d HMA Trend Filter

Hypothesis: The Ehlers Fisher Transform normalizes price to a Gaussian distribution,
making extreme values statistically meaningful for reversal entries. In bear/range
markets (like 2025), Fisher crossovers at extremes catch reversals better than RSI.

Why this might work:
- Fisher Transform has proven edge in mean-reverting markets (research shows 65-70% win rate)
- 1d HMA provides trend bias without being too restrictive (unlike ADX filters that killed #237)
- 4h timeframe balances signal quality vs trade frequency (avoided the noise of #235/#236)
- Fisher extremes (-1.5/+1.5) trigger more often than RSI<20/>80 = more trades
- Simple logic = fewer conditions that can conflict = actual trade execution

Learning from failures:
- #236 (30m Fisher): Sharpe=-9.213 - wrong TF (too noisy), no HTF filter
- #237 (1h KAMA+ADX): Sharpe=-0.474 - ADX too restrictive, killed trades
- #238 (4h CHOP+CRSI): Sharpe=-0.056 - too many regime conditions
- Complex multi-condition strategies = 0 trades or late entries
- Need SIMPLER entry logic with HTF bias only

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_reversal_1d_hma_atr_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) over period
    3. Scale to -0.99 to +0.99: 0.999 * (2 * normalized - 1)
    4. Fisher = 0.5 * ln((1 + scaled) / (1 - scaled))
    5. Signal line = previous Fisher value
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    Exit: Fisher crosses opposite direction or stoploss
    """
    n = len(close)
    fisher = np.zeros(n)
    signal = np.zeros(n)
    
    for i in range(period, n):
        # Typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Highest high and lowest low over lookback
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Normalize
        price_range = highest - lowest
        if price_range > 0:
            normalized = (typical - lowest) / price_range
        else:
            normalized = 0.5
        
        # Scale to -0.99 to +0.99
        scaled = 0.999 * (2.0 * normalized - 1.0)
        scaled = np.clip(scaled, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + scaled) / (1.0 - scaled))
        
        # Signal line is previous Fisher value
        signal[i] = fisher[i - 1] if i > 0 else 0.0
    
    return fisher, signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_idx = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Also check Fisher extreme levels for continuation
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # RSI filter to avoid false signals
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Fisher cross above -1.5 + RSI oversold + 1d bias not bearish
        if fisher_cross_long and rsi_oversold:
            if bull_trend_1d or not bear_trend_1d:
                new_signal = SIZE_BASE
        
        # Fisher extreme low (deep oversold) + 1d neutral/bullish
        elif fisher_extreme_low and rsi_oversold:
            if bull_trend_1d or not bear_trend_1d:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Fisher cross below +1.5 + RSI overbought + 1d bias not bullish
        if fisher_cross_short and rsi_overbought:
            if bear_trend_1d or not bull_trend_1d:
                new_signal = -SIZE_BASE
        
        # Fisher extreme high (deep overbought) + 1d neutral/bearish
        elif fisher_extreme_high and rsi_overbought:
            if bear_trend_1d or not bull_trend_1d:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.5 * atr[entry_idx]:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.5 * atr[entry_idx]:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === EXIT ON FISHER REVERSAL ===
        # If in long and Fisher goes above +1.0, consider exit
        if in_position and position_side > 0 and fisher[i] > 1.0:
            if new_signal == 0.0 or new_signal == SIZE_HALF:
                pass  # Already reducing
            else:
                new_signal = SIZE_HALF  # Reduce on Fisher reversal
        
        # If in short and Fisher goes below -1.0, consider exit
        if in_position and position_side < 0 and fisher[i] < -1.0:
            if new_signal == 0.0 or new_signal == -SIZE_HALF:
                pass  # Already reducing
            else:
                new_signal = -SIZE_HALF  # Reduce on Fisher reversal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_idx = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals