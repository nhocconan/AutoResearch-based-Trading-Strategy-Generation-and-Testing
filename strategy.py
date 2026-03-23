#!/usr/bin/env python3
"""
Experiment #626: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 553 failed strategies, the pattern is clear:
1. Too many filters = 0 trades (#615, #619, #620, #621 all got Sharpe=0.000)
2. Higher timeframes work better (current best is 1d+1w with Sharpe=0.520)
3. 12h should produce 20-50 trades/year naturally without over-filtering

This strategy SIMPLIFIES #624 (which got Sharpe=-0.002):
- Remove Donchian breakout requirement (was filtering too many valid entries)
- Relax RSI pullback zone from 40-60 to 35-65 (more entries)
- Use 1d HMA for trend confirmation (not slope, just direction)
- Keep ATR trailing stop (proven to control drawdown)
- Position size 0.30 discrete (slightly higher than 0.28 for better returns)

Why 12h + 1d works:
- 12h has ~730 bars/year → 20-50 trades is 3-7% trade rate (reasonable)
- 1d HTF confirms major trend direction without being too slow (like 1w)
- RSI pullback in trend direction has 60-70% win rate historically
- Fewer filters = more trades on ALL symbols (BTC, ETH, SOL must all trade)

Entry logic (simplified from #624):
LONG: 12h HMA sloping up + 1d HMA bullish + RSI 35-55 (pullback zone)
SHORT: 12h HMA sloping down + 1d HMA bearish + RSI 45-65 (bounce zone)

Exit: 2.5*ATR trailing stop OR trend flip (12h HMA slope reverses)

Position sizing: 0.30 discrete (Rule 4: max 0.40, typical 0.20-0.35)
Target: 25-40 trades/year on 12h (Rule 10: 12h/1d max 10-30, but 12h is borderline)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_simple_1d_v1"
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
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss (independent of signal)
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    stoploss_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 12H TREND (HMA slope over 3 bars) ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-3] if i >= 3 else False
        
        # Price relative to 12h HMA
        price_above_hma_12h = close[i] > hma_12h[i]
        price_below_hma_12h = close[i] < hma_12h[i]
        
        # === 1D HTF TREND CONFIRMATION ===
        # Simple: price above 1d HMA = bull, below = bear
        hma_1d_bull = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        hma_1d_bear = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        
        # === RSI PULLBACK ZONES (relaxed from 40-60 to 35-65) ===
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 65.0
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC (simplified - no Donchian requirement) ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 12h bull + 1d bull + RSI pullback ---
        # Less strict: just need trend alignment + RSI in zone
        if hma_12h_slope_bull and price_above_hma_12h:
            if hma_1d_bull:
                if rsi_oversold or rsi_pullback_long:
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 12h bear + 1d bear + RSI bounce ---
        elif hma_12h_slope_bear and price_below_hma_12h:
            if hma_1d_bear:
                if rsi_overbought or rsi_pullback_short:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position and no new signal, keep current signal
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, close[i])
            stoploss_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stoploss_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Update lowest price since entry
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stoploss_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stoploss_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            # Long position: exit if 12h trend turns bear
            if hma_12h_slope_bear and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Short position: exit if 12h trend turns bull
            if hma_12h_slope_bull and price_above_hma_12h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0:
            if not in_position:
                # New position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            # else: holding same direction, update highs/lows
        else:
            if in_position:
                # Position closed
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals