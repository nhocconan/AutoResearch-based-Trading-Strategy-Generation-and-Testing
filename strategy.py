#!/usr/bin/env python3
"""
Experiment #418: 4h EMA Pullback + 1d HMA Trend + RSI Momentum Filter

Hypothesis: After 417 failed experiments, the key insight is that COMPLEXITY is the enemy.
Strategies with too many filters (ADX + RSI + Donchian + Chop + Fisher) create mutually
exclusive conditions that generate 0 trades or fail on specific symbols.

This strategy uses SIMPLER logic that should work on BTC/ETH/SOL individually:

1. 1d HMA(21) TREND BIAS (via mtf_data helper):
   - Long only when price > 1d HMA (bullish bias)
   - Short only when price < 1d HMA (bearish bias)
   - This is the PRIMARY filter - no counter-trend trades

2. 4h EMA(21) PULLBACK ENTRY:
   - Long when price pulls back to EMA21 + bounces above it (in bull trend)
   - Short when price rallies to EMA21 + rejects below it (in bear trend)
   - Simpler than Donchian breakout, catches more entries

3. RSI(14) MOMENTUM CONFIRMATION (LOOSE thresholds):
   - Long: RSI > 40 (not oversold, has momentum)
   - Short: RSI < 60 (not overbought, has downward momentum)
   - These are WIDE thresholds to ensure we get trades

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from crashes while allowing normal volatility

5. POSITION SIZING: 0.25 discrete (conservative)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why this should work better than #413-#417:
- FEWER conditions = more trades (critical for Sharpe > 0)
- No ADX filter (ADX > 25 is too restrictive, kills trade count)
- No regime detection (adds complexity without edge)
- RSI thresholds are WIDE (40/60 vs 30/70) to generate signals
- 1d HMA is the only HTF filter (simpler than dual 1d+1w)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_ema_pullback_1d_hma_rsi_momentum_atr_v1"
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS (PRIMARY FILTER) ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h EMA PULLBACK ENTRY ===
        # Long: price was below EMA, now crosses above (pullback bounce)
        ema_long = close[i] > ema_21[i] and close[i-1] <= ema_21[i-1]
        # Short: price was above EMA, now crosses below (pullback rejection)
        ema_short = close[i] < ema_21[i] and close[i-1] >= ema_21[i-1]
        
        # === RSI MOMENTUM CONFIRMATION (WIDE THRESHOLDS) ===
        # Long: RSI > 40 (has upward momentum, not dying)
        rsi_long = rsi[i] > 40
        # Short: RSI < 60 (has downward momentum, not rallying)
        rsi_short = rsi[i] < 60
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: 1d bull trend + 4h EMA pullback bounce + RSI momentum
        if bull_trend_1d and ema_long and rsi_long:
            new_signal = SIZE
        
        # SHORT: 1d bear trend + 4h EMA pullback rejection + RSI momentum
        elif bear_trend_1d and ema_short and rsi_short:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d trend turns bear
        if in_position and position_side > 0 and bear_trend_1d:
            new_signal = 0.0
        
        # Exit short if 1d trend turns bull
        if in_position and position_side < 0 and bull_trend_1d:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals