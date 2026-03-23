#!/usr/bin/env python3
"""
Experiment #047: 1d Primary + 1w HTF — Donchian Breakout with HMA Trend Filter

Hypothesis: Based on experiment history showing 1d strategies (#037, #039, #043) achieved
positive Sharpe with Donchian+HMA+RSI combinations, I'm simplifying the entry logic to
ensure sufficient trade generation while maintaining quality filters.

Key innovations:
1. DONCHIAN BREAKOUT: 20-day high/low breakout — proven trend entry on daily
2. HMA(21) trend filter — only trade in direction of HMA slope
3. 1w HMA for macro bias — align with weekly trend
4. RSI(14) momentum confirmation — avoid entering at extremes against trend
5. ATR(14) trailing stop — 2.5*ATR for risk management

Why 1d works:
- Targets 20-50 trades/year (Rule 10 compliant)
- Less noise, cleaner signals than lower TF
- Proven in experiments #037, #039, #043 (all positive Sharpe on 1d)

Entry conditions (LOOSE enough to generate trades):
- Long: Donchian breakout + HMA sloping up + RSI > 45 + price > 1w HMA
- Short: Donchian breakout + HMA sloping down + RSI < 55 + price < 1w HMA

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_v1"
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
    """Calculate RSI."""
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
    """Calculate Donchian Channel (20-day high/low)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(donchian_high[i]) or np.isnan(hma_1d[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D HMA TREND ===
        hma_1d_slope_bull = hma_1d[i] > hma_1d[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d[i] < hma_1d[i-5] if i >= 5 else False
        price_above_hma_1d = close[i] > hma_1d[i]
        price_below_hma_1d = close[i] < hma_1d[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_high[i-1]  # Break above previous 20-day high
        breakout_low = close[i] < donchian_low[i-1]   # Break below previous 20-day low
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 45.0  # Not oversold for long entries
        rsi_bearish = rsi_14[i] < 55.0  # Not overbought for short entries
        rsi_strong_bull = rsi_14[i] > 50.0
        rsi_strong_bear = rsi_14[i] < 50.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entry: Donchian breakout + HMA bullish + RSI confirmation + 1w bias
        if breakout_high and hma_1d_slope_bull and rsi_bullish:
            if price_above_hma_1w or price_above_hma_1d:  # At least one trend confirmation
                new_signal = POSITION_SIZE
        
        # Short entry: Donchian breakout + HMA bearish + RSI confirmation + 1w bias
        if breakout_low and hma_1d_slope_bear and rsi_bearish:
            if price_below_hma_1w or price_below_hma_1d:  # At least one trend confirmation
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
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if HMA turns bearish
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if HMA turns bullish
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
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