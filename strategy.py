#!/usr/bin/env python3
"""
Experiment #249: 1h Supertrend + RSI Pullback + 4h HMA Trend Filter

Hypothesis: 1h timeframe with strong 4h trend filter can capture intraday swings
while avoiding whipsaws. Using Supertrend for trend direction + RSI pullback
entries + 4h HMA for higher timeframe bias.

Why this might work on 1h:
- 1h captures more intraday moves than 4h/12h
- Supertrend provides clear trend direction with ATR-based stops
- RSI pullback (not extreme) entries catch trend continuations
- 4h HMA filter prevents trading against major trend
- Simpler than failed 1h strategies (#237, #241, #243) - fewer conflicting filters
- Looser RSI thresholds (35/65) ensure enough trades

Key differences from failed 1h strategies:
- #237 used KAMA+ADX (too many indicators, conflicting signals)
- #241 used 15m primary with volume (too noisy, volume unreliable)
- #243 used Z-score regime (mean reversion failed on 1h)
- This uses cleaner Supertrend + RSI pullback (proven combination)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: Supertrend provides built-in stop, plus 2*ATR emergency stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_rsi_pullback_4h_hma_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_line, supertrend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize final bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    # First valid bar
    if period < n:
        final_upper[period] = upper_band[period]
        final_lower[period] = lower_band[period]
        supertrend[period] = upper_band[period]
        direction[period] = 1  # Start bullish by default
        
        # Calculate remaining bars
        for i in range(period + 1, n):
            # Update upper band
            if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i-1]
            
            # Update lower band
            if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i-1]
            
            # Determine supertrend and direction
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] > final_upper[i]:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1
            else:
                if close[i] < final_lower[i]:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
    
    # Fill initial values
    supertrend[:period] = np.nan
    direction[:period] = 0
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend_line, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    rsi_14 = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_price_idx = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend_line[i]) or np.isnan(supertrend_dir[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (4h HMA) ===
        # Only trade in direction of 4h trend
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        st_bullish = supertrend_dir[i] == 1
        st_bearish = supertrend_dir[i] == -1
        
        # === RSI PULLBACK DETECTION ===
        # For longs: RSI pulled back but not oversold (35-50 range in uptrend)
        # For shorts: RSI rallied but not overbought (50-65 range in downtrend)
        rsi_pullback_long = 35 <= rsi_14[i] <= 55
        rsi_pullback_short = 45 <= rsi_14[i] <= 65
        
        # === EMA TREND CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 4h bullish + Supertrend bullish + RSI pullback + EMA bullish ---
        if bull_trend_4h and st_bullish and rsi_pullback_long and ema_bullish:
            # Additional confirmation: price above EMA21
            if close[i] > ema_21[i]:
                new_signal = SIZE_BASE
        
        # --- SHORT ENTRY: 4h bearish + Supertrend bearish + RSI pullback + EMA bearish ---
        if bear_trend_4h and st_bearish and rsi_pullback_short and ema_bearish:
            # Additional confirmation: price below EMA21
            if close[i] < ema_21[i]:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Supertrend provides built-in stop, but add emergency ATR stop
        if in_position and position_side != 0:
            stoploss_triggered = False
            
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                
                # Supertrend stop: price crosses below supertrend line
                if close[i] < supertrend_line[i]:
                    stoploss_triggered = True
                
                # Emergency ATR stop: 2.5 * ATR below entry
                atr_stop = entry_price - 2.5 * atr[entry_price_idx]
                if close[i] < atr_stop:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                
                # Supertrend stop: price crosses above supertrend line
                if close[i] > supertrend_line[i]:
                    stoploss_triggered = True
                
                # Emergency ATR stop: 2.5 * ATR above entry
                atr_stop = entry_price + 2.5 * atr[entry_price_idx]
                if close[i] > atr_stop:
                    stoploss_triggered = True
            
            if stoploss_triggered:
                new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                risk = atr[entry_price_idx]
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                risk = atr[entry_price_idx]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position and signals[i-1] != 0.0:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals