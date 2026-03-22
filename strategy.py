#!/usr/bin/env python3
"""
Experiment #409: 15m Trend-Pullback System with 4h/1h HTF Filters

Hypothesis: After 408 failed experiments, the pattern is clear - complex regime 
switching overfits and fails. Simple trend-following on 15m gets whipsawed.
The solution: SIMPLE multi-timeframe alignment with pullback entries.

STRATEGY COMPONENTS:
1. 4h HMA(21) TREND BIAS: Major trend direction
   - Only long when price > 4h HMA
   - Only short when price < 4h HMA
   - HMA smoother than EMA, less lag

2. 1h RSI(14) MOMENTUM FILTER: Confirms trend strength
   - Long only when 1h RSI > 45 (bullish momentum)
   - Short only when 1h RSI < 55 (bearish momentum)
   - Avoids entering against HTF momentum

3. 15m RSI(7) PULLBACK ENTRY: Timing entries on dips
   - Long when 15m RSI(7) crosses above 35 from below (pullback end)
   - Short when 15m RSI(7) crosses below 65 from above (rally end)
   - Looser thresholds than extreme mean-reversion (20/80) to generate trades

4. ATR(14) TRAILING STOP: Risk management
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from crashes while allowing trend runs

5. POSITION SIZING: 0.30 discrete
   - 30% capital per position (conservative for 15m volatility)
   - Discrete levels minimize fee churn

Why this should work on 15m:
- 4h HMA filters out 15m noise and whipsaw
- 1h RSI confirms momentum alignment across timeframes
- 15m RSI pullback entries catch trend continuations (not reversals)
- Should generate 50-100 trades/year (enough for stats, not too many for fees)
- Works in both bull (2021) and bear (2022, 2025) markets
- Simple logic = less overfitting than complex regime systems

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h and 4h via mtf_data helper (call ONCE before loop each)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_pullback_4h_hma_1h_rsi_atr_v1"
timeframe = "15m"
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
    """Calculate RSI using standard Wilder's method."""
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
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 7)  # Faster RSI for pullback detection
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Track RSI crosses
    prev_rsi_15m = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            prev_rsi_15m = rsi_15m[i] if not np.isnan(rsi_15m[i]) else prev_rsi_15m
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            prev_rsi_15m = rsi_15m[i] if not np.isnan(rsi_15m[i]) else prev_rsi_15m
            continue
        
        if np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            prev_rsi_15m = rsi_15m[i] if not np.isnan(rsi_15m[i]) else prev_rsi_15m
            continue
        
        if np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            prev_rsi_15m = rsi_15m[i] if not np.isnan(rsi_15m[i]) else prev_rsi_15m
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h RSI MOMENTUM FILTER ===
        bull_momentum_1h = rsi_1h_aligned[i] > 45.0
        bear_momentum_1h = rsi_1h_aligned[i] < 55.0
        
        # === 15m RSI PULLBACK ENTRY ===
        rsi_cross_above_35 = (prev_rsi_15m <= 35.0 and rsi_15m[i] > 35.0)
        rsi_cross_below_65 = (prev_rsi_15m >= 65.0 and rsi_15m[i] < 65.0)
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: 4h uptrend + 1h bullish momentum + 15m pullback end
        if bull_trend_4h and bull_momentum_1h and rsi_cross_above_35:
            new_signal = SIZE
        
        # SHORT: 4h downtrend + 1h bearish momentum + 15m rally end
        elif bear_trend_4h and bear_momentum_1h and rsi_cross_below_65:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0 and new_signal != 0.0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            elif position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
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
        prev_rsi_15m = rsi_15m[i]
    
    return signals