#!/usr/bin/env python3
"""
Experiment #619: 4h Primary + 1d HTF — HMA Trend + Donchian Breakout + RSI Pullback

Hypothesis: After 547 failed strategies, the pattern is clear — regime-switching adds 
complexity without benefit. Strategies #614 (Sharpe=0.089) and literature patterns 
(Donchian+HMA+RSI for SOL Sharpe=+0.782, HMA+RSI+ATR for SOL Sharpe=+0.879) suggest 
a simpler trend-following approach works better.

Key insights from failures:
1. Regime-switching (CHOP-based) consistently fails — #607, #609, #611, #617 all negative
2. Too many filters = 0 trades or overfitting
3. HMA is faster than EMA for trend detection (proven in literature)
4. Donchian breakouts capture momentum well without whipsaw
5. RSI pullback entries (not extremes) work better in trends

Strategy logic:
- 1d HMA slope = primary trend direction (HTF filter)
- 4h Donchian(20) breakout = entry trigger
- 4h RSI(14) 40-60 = pullback entry (not chasing extremes)
- 2.5*ATR trailing stop = risk management
- Discrete position size 0.30

Why this might beat Sharpe=0.520:
- Simpler = less overfitting (lesson from 547 failures)
- Donchian breakout proven on SOL (Sharpe +0.782)
- HMA faster than KAMA for trend detection
- RSI pullback (40-60) avoids extreme entries that fail in trends
- 1d HTF filter keeps us on right side of major moves
- Target 25-45 trades/year on 4h (per Rule 10)

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_rsi_1d_v1"
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
    Much faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    # HMA formula
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (HMA slope over 3 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H MOMENTUM (HMA slope 2 bars) ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-2] if i >= 2 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-2] if i >= 2 else False
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout up: close crosses above upper band
        donchian_breakout_up = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        # Breakout down: close crosses below lower band
        donchian_breakout_down = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # === RSI PULLBACK FILTER (not extremes) ===
        # In uptrend, want RSI pullback to 40-55 (not oversold <30)
        rsi_ok_long = 40.0 <= rsi_14[i] <= 60.0
        # In downtrend, want RSI bounce to 45-65 (not overbought >70)
        rsi_ok_short = 40.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # LONG: 1d bull trend + 4h momentum + Donchian breakout + RSI pullback
        if hma_1d_slope_bull and price_above_hma_1d and hma_4h_slope_bull:
            if donchian_breakout_up and rsi_ok_long:
                new_signal = POSITION_SIZE
        
        # SHORT: 1d bear trend + 4h momentum + Donchian breakout + RSI bounce
        elif hma_1d_slope_bear and price_below_hma_1d and hma_4h_slope_bear:
            if donchian_breakout_down and rsi_ok_short:
                new_signal = -POSITION_SIZE
        
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            # Exit long if 1d trend flips bear
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend flips bull
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0:
            if not in_position:
                # New entry
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
            # else: same direction, keep tracking
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