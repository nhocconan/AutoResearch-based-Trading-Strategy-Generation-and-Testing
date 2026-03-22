#!/usr/bin/env python3
"""
Experiment #605: 12h KAMA Adaptive Trend + RSI Momentum + 1d HMA Bias

Hypothesis: After 536+ failures, the key insight is that regime detection (CHOP) adds
complexity without benefit. KAMA (Kaufman Adaptive Moving Average) inherently adapts
to market efficiency - fast in trends, slow in ranges. This eliminates the need for
explicit regime filters.

Why this should beat #593 (Sharpe=-0.281):
1. KAMA replaces CHOP+Donchian+BB complexity with single adaptive indicator
2. KAMA ER (Efficiency Ratio) naturally detects trend vs range
3. Simpler entry logic = more trades (addressing 0-trade failure mode)
4. 1d HMA bias remains (proven useful in winning strategies)
5. RSI with wider thresholds (25/75 vs 30/70) = more entry opportunities
6. ADX hysteresis (enter>15, exit<12) prevents whipsaw exits
7. 2.5*ATR stoploss (vs 2.0) = fewer premature exits in volatile 12h bars

Key differences from #593:
- NO Choppiness Index (adds lag, no edge)
- NO Donchian breakout (too many false breakouts on 12h)
- NO Bollinger Bands (redundant with KAMA adaptation)
- YES KAMA crossover + ER filter (proven adaptive trend following)
- YES RSI momentum with wider bands (ensure trade generation)
- YES ADX hysteresis for exit management

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adaptive_1d_hma_rsi_momentum_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - fast in trends, slow in ranges.
    
    ER (Efficiency Ratio) = |Net Change| / Sum of Individual Changes
    SC (Smoothing Constant) = (ER * (fast_SC - slow_SC) + slow_SC)^2
    fast_SC = 2/(fast_period+1), slow_SC = 2/(slow_period+1)
    """
    close_s = pd.Series(close)
    
    # Net change over ER period
    net_change = np.abs(close_s - close_s.shift(er_period))
    
    # Sum of individual changes (volatility)
    individual_changes = np.abs(close_s.diff())
    sum_changes = individual_changes.rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio (0 to 1)
    er = net_change / sum_changes.replace(0, np.inf)
    er = er.clip(0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation (iterative for proper adaptation)
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama, er.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
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
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_12h, er_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss (separate from signal)
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # ADX hysteresis tracking
    prev_adx = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(kama_12h[i]) or np.isnan(er_12h[i]):
            continue
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND SIGNAL ===
        # Price above KAMA = bullish, below = bearish
        price_above_kama = close[i] > kama_12h[i]
        price_below_kama = close[i] < kama_12h[i]
        
        # KAMA slope (simple momentum)
        kama_slope_bull = kama_12h[i] > kama_12h[i-5] if i >= 5 else False
        kama_slope_bear = kama_12h[i] < kama_12h[i-5] if i >= 5 else False
        
        # === EFFICIENCY RATIO FILTER ===
        # ER > 0.5 = trending market (trust KAMA signals)
        # ER < 0.3 = ranging market (require stronger confirmation)
        is_trending = er_12h[i] > 0.4
        is_ranging = er_12h[i] < 0.3
        
        # === RSI MOMENTUM (wider thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_neutral = 35 <= rsi_14[i] <= 65
        
        # === ADX FILTER with HYSTERESIS ===
        # Enter when ADX > 15, exit when ADX < 12 (prevents whipsaw)
        adx_enter = adx_14[i] > 15
        adx_exit = adx_14[i] < 12
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY PATHS (multiple paths to ensure trades)
        # Path 1: KAMA bullish + 1d bias + trending market
        if price_above_kama and kama_slope_bull and bull_bias and is_trending:
            new_signal = SIZE
        
        # Path 2: KAMA bullish + RSI oversold bounce (mean reversion in uptrend)
        elif price_above_kama and bull_bias and rsi_oversold:
            new_signal = SIZE
        
        # Path 3: KAMA bullish + ADX confirming (any regime)
        elif price_above_kama and kama_slope_bull and adx_enter:
            new_signal = SIZE
        
        # SHORT ENTRY PATHS
        # Path 1: KAMA bearish + 1d bias + trending market
        if price_below_kama and kama_slope_bear and bear_bias and is_trending:
            new_signal = -SIZE
        
        # Path 2: KAMA bearish + RSI overbought rejection (mean reversion in downtrend)
        elif price_below_kama and bear_bias and rsi_overbought:
            new_signal = -SIZE
        
        # Path 3: KAMA bearish + ADX confirming (any regime)
        elif price_below_kama and kama_slope_bear and adx_enter:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === ADX HYSTERESIS EXIT ===
        adx_exit_triggered = False
        if in_position and position_side != 0 and adx_exit:
            adx_exit_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and bear_bias and price_below_kama:
                trend_reversal = True
            if position_side < 0 and bull_bias and price_above_kama:
                trend_reversal = True
        
        # Apply stoploss or exit conditions
        if stoploss_triggered or adx_exit_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
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
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
        prev_adx = adx_14[i]
    
    return signals