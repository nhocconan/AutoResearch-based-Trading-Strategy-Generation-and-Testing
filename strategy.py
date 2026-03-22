#!/usr/bin/env python3
"""
Experiment #616: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: Building on proven 12h patterns (Donchian+HMA+RSI achieved SOL Sharpe +0.782),
this strategy combines Donchian channel breakouts with HMA trend filtering and RSI
momentum confirmation. The 1d HTF provides major trend bias to avoid counter-trend trades.

Why this might beat Sharpe=0.520:
1. Donchian(20) breakouts capture sustained moves without whipsaw of EMA crossovers
2. HMA(21/50) crossover confirms trend direction with less lag than SMA
3. RSI(14) filter avoids entering at momentum extremes (RSI 35-65 sweet spot)
4. 1d HMA slope provides major trend bias (only trade with HTF trend)
5. ATR(14) trailing stop (2.5*ATR) limits losses on false breakouts
6. Conservative size (0.30) controls drawdown through volatile periods

Key differences from failed #607 (1d KAMA+CHOP+RSI, Sharpe=-0.627):
- Simpler logic (no complex regime-switching that kills trade frequency)
- Donchian breakout vs KAMA slope (more reliable entry trigger)
- 12h vs 1d (higher TF = fewer false signals, proven in experiment history)
- HMA vs KAMA (HMA has less lag, better for breakout confirmation)

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 20-50 trades/year on 12h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_rsi_1d_v1"
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
    Reduces lag while maintaining smoothness.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel upper and lower bounds.
    Upper = highest high over N periods
    Lower = lowest low over N periods
    """
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
    
    # Calculate 1d HMA for major trend direction
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
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
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        if np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D HTF TREND BIAS (HMA slope over 3 bars) ===
        hma_1d_slope_bull = False
        hma_1d_slope_bear = False
        if i >= 3 and not np.isnan(hma_1d_50_aligned[i-3]):
            hma_1d_slope_bull = hma_1d_50_aligned[i] > hma_1d_50_aligned[i-3]
            hma_1d_slope_bear = hma_1d_50_aligned[i] < hma_1d_50_aligned[i-3]
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_50_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_50_aligned[i]
        
        # === 12H HMA CROSSOVER ===
        hma_cross_bull = hma_12h_21[i] > hma_12h_50[i]
        hma_cross_bear = hma_12h_21[i] < hma_12h_50[i]
        
        # HMA slope (2 bars)
        hma_12h_slope_bull = False
        hma_12h_slope_bear = False
        if i >= 2 and not np.isnan(hma_12h_21[i-2]):
            hma_12h_slope_bull = hma_12h_21[i] > hma_12h_21[i-2]
            hma_12h_slope_bear = hma_12h_21[i] < hma_12h_21[i-2]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI MOMENTUM FILTER (avoid extremes) ===
        rsi_neutral = 35.0 <= rsi_14[i] <= 65.0
        rsi_bullish = 40.0 <= rsi_14[i] <= 70.0
        rsi_bearish = 30.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: 1d bull bias + 12h HMA bull + Donchian breakout + RSI confirmation
        if hma_1d_slope_bull and price_above_hma_1d:
            if hma_cross_bull and hma_12h_slope_bull:
                if donchian_breakout_long and rsi_bullish:
                    new_signal = POSITION_SIZE
        
        # SHORT: 1d bear bias + 12h HMA bear + Donchian breakdown + RSI confirmation
        elif hma_1d_slope_bear and price_below_hma_1d:
            if hma_cross_bear and hma_12h_slope_bear:
                if donchian_breakout_short and rsi_bearish:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_cross_bear or (hma_1d_slope_bear and price_below_hma_1d):
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_cross_bull or (hma_1d_slope_bull and price_above_hma_1d):
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