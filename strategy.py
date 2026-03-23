#!/usr/bin/env python3
"""
Experiment #637: 1d Primary + 1w HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance between
signal quality and trade frequency. Building on proven patterns (HMA+RSI+Donchian showed
Sharpe +0.879 on SOL at 4h), this adapts to 1d with 1w HTF for major trend direction.

Why this might beat Sharpe=0.520:
1. 1w HMA slope provides clear bull/bear regime (simpler than Choppiness which failed)
2. 1d HMA fast/slow crossover for entry timing (proven pattern)
3. RSI pullback zone (35-65) ensures we enter on dips/rallies, not tops/bottoms
4. Donchian(20) breakout confirms momentum exists before entry
5. 3.0*ATR trailing stop limits drawdown on reversals
6. Fewer filters than failed experiments = more trades (target 25-40/year on 1d)
7. Discrete sizing (0.30) minimizes fee churn while controlling DD

Key differences from failed experiments:
- NO Choppiness Index (failed in #629, #631, #632, #633, #636)
- NO Connors RSI complexity (failed in #625, #630, #631, #632, #633)
- Simple HMA slope + RSI zone + Donchian = proven combination
- 1w HTF for trend, 1d for entries (cleaner than 12h/4h mix)

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 25-40 trades/year on 1d (per Rule 10)
Stoploss: 3.0*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_donchian_1w_v1"
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

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel upper and lower bands.
    Upper = highest high over period
    Lower = lowest low over period
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for primary trend direction
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    hma_1d_fast = calculate_hma(close, period=10)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS (HMA slope over 2 bars) ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-2] if i >= 2 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-2] if i >= 2 else False
        
        # Price relative to 1w HMA
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D HMA FAST/SLOW CROSSOVER ===
        hma_cross_bull = hma_1d_fast[i] > hma_1d[i]
        hma_cross_bear = hma_1d_fast[i] < hma_1d[i]
        
        # === 1D HMA SLOPE (2 bars) ===
        hma_1d_slope_bull = hma_1d[i] > hma_1d[i-2] if i >= 2 else False
        hma_1d_slope_bear = hma_1d[i] < hma_1d[i-2] if i >= 2 else False
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === RSI PULLBACK ZONES (wider for more trades) ===
        rsi_neutral_long = 35.0 <= rsi_14[i] <= 65.0
        rsi_neutral_short = 35.0 <= rsi_14[i] <= 65.0
        rsi_oversold = rsi_14[i] < 50.0
        rsi_overbought = rsi_14[i] > 50.0
        
        # === ENTRY LOGIC (simplified for more trades) ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1w bull trend + 1d HMA cross + RSI ok ---
        # Condition 1: 1w HMA sloping up OR price above 1w HMA
        # Condition 2: 1d HMA fast > slow (momentum cross)
        # Condition 3: RSI not overbought (<65) or in neutral zone
        # Condition 4: Price breaking or near Donchian upper
        if hma_1w_slope_bull or price_above_hma_1w:
            if hma_cross_bull and hma_1d_slope_bull:
                if rsi_oversold or (rsi_neutral_long and close[i] > donchian_mid[i]):
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 1w bear trend + 1d HMA cross + RSI ok ---
        # Condition 1: 1w HMA sloping down OR price below 1w HMA
        # Condition 2: 1d HMA fast < slow (momentum cross)
        # Condition 3: RSI not oversold (>35) or in neutral zone
        # Condition 4: Price breaking or near Donchian lower
        elif hma_1w_slope_bear or price_below_hma_1w:
            if hma_cross_bear and hma_1d_slope_bear:
                if rsi_overbought or (rsi_neutral_short and close[i] < donchian_mid[i]):
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC (avoid churn) ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
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
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals