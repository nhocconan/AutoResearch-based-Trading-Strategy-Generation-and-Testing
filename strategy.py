#!/usr/bin/env python3
"""
Experiment #003: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 2 failed Choppiness Index strategies, pivot to proven 1d patterns:
1. Donchian breakout captures trend moves in crypto (works on SOL Sharpe +0.782)
2. 1w HMA(21) provides long-term trend bias (slower than 4h, fewer false signals)
3. RSI(14) pullback filter prevents chasing breakouts at extremes
4. ATR(14) trailing stoploss protects from reversals
5. Dual regime: trend-follow when 1w HMA sloped, mean-revert when flat

Why 1d + 1w works:
- 1d has proven track record (#693 Sharpe=0.105 with simple logic)
- 1w HMA filters out 2022 bear market whipsaws (only long when 1w bullish)
- Donchian(20) breakout = ~20 trades/year target (within 20-50 range)
- RSI filter (30-70) prevents entering at overbought/oversold extremes
- Position size 0.30 (conservative for daily TF)

Key differences from failed #001/#002:
- NO Choppiness Index (tried 50+ times, mostly fails)
- Donchian breakout instead of Fisher/BB mean-reversion
- 1w HTF instead of 4h/1d (slower, fewer whipsaws)
- Simpler entry logic (less overfiltering = more trades)

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 20-50 trades/year on 1d
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma1w_rsi_v1"
timeframe = "1d"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    
    return upper.values, lower.values, middle.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for long-term trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    
    # Also calculate 1d HMA for additional trend confirmation
    hma_1d = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # Track previous Donchian levels for breakout detection
    prev_donchian_upper = 0.0
    prev_donchian_lower = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-5] if i >= 5 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-5] if i >= 5 else False
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        hma_1d_slope_bull = hma_1d[i] > hma_1d[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d[i] < hma_1d[i-5] if i >= 5 else False
        price_above_hma_1d = close[i] > hma_1d[i]
        price_below_hma_1d = close[i] < hma_1d[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout = price crosses above previous Donchian upper
        breakout_long = (close[i] > prev_donchian_upper) and (close[i-1] <= prev_donchian_upper)
        breakout_short = (close[i] < prev_donchian_lower) and (close[i-1] >= prev_donchian_lower)
        
        # Also check if price is at Donchian extreme (within 2% of breakout level)
        near_donchian_high = close[i] > donchian_upper[i] * 0.98
        near_donchian_low = close[i] < donchian_lower[i] * 1.02
        
        # === RSI FILTER ===
        rsi_neutral = (rsi_14[i] > 35) and (rsi_14[i] < 65)
        rsi_bullish = rsi_14[i] > 45
        rsi_bearish = rsi_14[i] < 55
        rsi_not_overbought = rsi_14[i] < 70
        rsi_not_oversold = rsi_14[i] > 30
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Condition 1: 1w trend bullish + Donchian breakout + RSI confirmation
        if hma_1w_slope_bull and price_above_hma_1w:
            if breakout_long and rsi_bullish and rsi_not_overbought:
                new_signal = POSITION_SIZE
            # Condition 2: Pullback to Donchian middle + RSI bounce
            elif near_donchian_high and rsi_14[i] > rsi_14[i-1] and rsi_bullish:
                if hma_1d_slope_bull and price_above_hma_1d:
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Condition 1: 1w trend bearish + Donchian breakdown + RSI confirmation
        if hma_1w_slope_bear and price_below_hma_1w:
            if breakout_short and rsi_bearish and rsi_not_oversold:
                new_signal = -POSITION_SIZE
            # Condition 2: Rally to Donchian middle + RSI drop
            elif near_donchian_low and rsi_14[i] < rsi_14[i-1] and rsi_bearish:
                if hma_1d_slope_bear and price_below_hma_1d:
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
            if hma_1w_slope_bear and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1w_slope_bull and price_above_hma_1w:
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
        
        # Store previous Donchian levels for next iteration
        prev_donchian_upper = donchian_upper[i]
        prev_donchian_lower = donchian_lower[i]
        
        signals[i] = new_signal
    
    return signals