#!/usr/bin/env python3
"""
Experiment #640: 1h Primary + 4h/12h HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 566 failed strategies, the pattern is clear:
- Too many filters = 0 trades (#632, #635, #638 all Sharpe=0.000)
- Complex regime switching = paralysis (#629, #631, #636 negative Sharpe)
- Lower TF (1h/30m) needs SIMPLER logic than 4h/12h/1d

This strategy uses MINIMAL confluence for 1h:
1. 12h HMA slope = trend direction (1 filter, not 3)
2. 1h RSI pullback = entry timing (35-55 long, 45-65 short)
3. 1h ATR stoploss = risk management (2.5*ATR)
4. NO session filter (kills trades)
5. NO volume filter (kills trades)
6. NO choppiness index (failed in #629, #631, #636)

Key insight from failures: 1h needs FEWER filters than 4h, not more.
The 12h HMA provides the "smart money" direction, 1h RSI provides entry.
This should generate 40-60 trades/year (within 30-80 target) with positive Sharpe.

Position sizing: 0.25 (conservative for 1h, per Rule 4 max 0.40)
Stoploss: 2.5*ATR trailing (proven in baseline strategies)
Target: Beat Sharpe=0.520 from mtf_1d_chop_crsi_regime_1w_v1
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_simple_12h_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for primary trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1h[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (HMA slope over 3 bars) ===
        # Simple: is 12h HMA higher than 3 bars ago?
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-3]
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-3]
        
        # Price relative to 12h HMA (confirmation)
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 1H RSI PULLBACK ZONES (simplified from failed experiments) ===
        # Long: RSI 35-55 (pullback in uptrend, not oversold which = crash)
        # Short: RSI 45-65 (bounce in downtrend, not overbought which = rally)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 65.0
        
        # === 1H HMA SLOPE (momentum confirmation, 2 bars) ===
        hma_1h_slope_bull = hma_1h[i] > hma_1h[i-2]
        hma_1h_slope_bear = hma_1h[i] < hma_1h[i-2]
        
        # === ENTRY LOGIC (MINIMAL confluence for trade frequency) ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 12h bull trend + 1h RSI pullback + 1h HMA momentum ---
        # Only 3 filters (not 5+ which caused 0 trades in #635)
        if hma_12h_slope_bull and price_above_hma_12h:
            if rsi_pullback_long and hma_1h_slope_bull:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 12h bear trend + 1h RSI bounce + 1h HMA momentum ---
        elif hma_12h_slope_bear and price_below_hma_12h:
            if rsi_pullback_short and hma_1h_slope_bear:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If we're in a position and no new signal, hold the position
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP (12h HMA reverses) ===
        if in_position and position_side > 0:
            if hma_12h_slope_bear and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_12h_slope_bull and price_above_hma_12h:
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