#!/usr/bin/env python3
"""
Experiment #491: 12h Dual HMA Crossover with Daily Trend Filter

Hypothesis: After 490 failed experiments, the critical insight is that 12h timeframe
needs SIMPLER logic with fewer conflicting filters. Complex regime filters (CHOP, ADX,
Z-score combinations) have caused 0 trades or negative Sharpe. This strategy uses:

1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Long only when price > 1d HMA (bullish bias)
   - Short only when price < 1d HMA (bearish bias)
   - Simple directional filter, no complex regime detection

2. 12h DUAL HMA CROSSOVER (8/21):
   - Fast HMA(8) crosses above Slow HMA(21) = long signal
   - Fast HMA(8) crosses below Slow HMA(21) = short signal
   - HMA reduces lag vs EMA, better for 12h swings

3. RSI(14) PULLBACK CONFIRMATION (loose threshold):
   - Long: RSI > 45 (not oversold, confirms momentum)
   - Short: RSI < 55 (not overbought, confirms downside)
   - Loose thresholds ensure sufficient trades

4. ATR(14) TRAILING STOP at 2.5x:
   - Appropriate for 12h volatility
   - Signal → 0 when price moves 2.5*ATR against position

5. POSITION SIZING: 0.25 discrete
   - Conservative for 12h swings
   - Discrete levels minimize fee churn

Why this should work on 12h:
- Simpler logic = more trades (critical for Sharpe calculation)
- Daily HMA provides robust trend filter without whipsaw
- Dual HMA crossover catches sustained moves on 12h
- Loose RSI filter ensures 20-40 trades/year per symbol
- Should generate ≥10 trades on train, ≥3 on test

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_hma_daily_trend_rsi_pullback_atr_v1"
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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    prev_hma_fast = 0.0
    prev_hma_slow = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_trend = close[i] > hma_1d_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i]
        
        # === HMA CROSSOVER DETECTION ===
        hma_cross_long = (hma_fast[i] > hma_slow[i]) and (prev_hma_fast <= prev_hma_slow)
        hma_cross_short = (hma_fast[i] < hma_slow[i]) and (prev_hma_fast >= prev_hma_slow)
        
        # === ENTRY LOGIC (simple, loose filters for sufficient trades) ===
        new_signal = 0.0
        
        # Long entry: HMA cross + bull trend + RSI confirmation
        if hma_cross_long and bull_trend and rsi[i] > 45:
            new_signal = SIZE
        
        # Short entry: HMA cross + bear trend + RSI confirmation
        if hma_cross_short and bear_trend and rsi[i] < 55:
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
        # Exit if daily trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend:
                new_signal = 0.0
            if position_side < 0 and bull_trend:
                new_signal = 0.0
        
        # === HMA CROSS REVERSAL EXIT ===
        # Exit long if HMA crosses bearish, exit short if HMA crosses bullish
        if in_position and position_side > 0 and hma_cross_short:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_cross_long:
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
        
        # Store previous HMA values for crossover detection
        prev_hma_fast = hma_fast[i]
        prev_hma_slow = hma_slow[i]
    
    return signals