#!/usr/bin/env python3
"""
Experiment #160: 1h Primary + 4h/12h HTF — Fisher Transform + ADX Regime Strategy

Hypothesis: Previous 1h strategies failed because RSI/CRSI are lagging indicators that
generate too many false signals in bear/range markets (2025 test period). This strategy
uses Ehlers Fisher Transform which is leading (not lagging) and catches reversals earlier.

Key innovations:
1) 12h HMA(21) for MACRO trend — slowest, most reliable bias filter
2) 4h HMA(21) for INTERMEDIATE trend — confirms 12h direction
3) Fisher Transform(9) for ENTRY — leading indicator, crosses -1.5/+1.5 for signals
4) ADX(14) for STRENGTH — only enter when ADX > 18 (avoid dead chop)
5) Volume > 0.6x avg — relaxed filter (previous 0.7-0.8x killed trade count)
6) ATR(14) stoploss at 2.5x — mandatory risk management
7) Hold logic — stay in position until Fisher reverses (not just exit at neutral)

Why Fisher over RSI/CRSI:
- Fisher is leading (transforms price to Gaussian distribution)
- Catches reversals 1-3 bars earlier than RSI
- Works better in bear markets (2025 test) and choppy conditions
- Proven in academic literature (Ehlers 2002)

Position sizing: 0.25 base, 0.35 with full HTF confluence
Target: 40-70 trades/year per symbol (fewer than RSI strategies = less fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_adx_hma_4h12h_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - leading indicator for reversals.
    Transforms price to Gaussian distribution for clearer signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price to 0-1 range
    price_range = highest - lowest
    normalized = np.zeros(len(close))
    mask = price_range > 0
    normalized[mask] = 0.999 * (hl2[mask] - lowest[mask]) / price_range[mask] + 0.001
    
    # Fisher transform
    fisher = np.zeros(len(close))
    fisher_prev = np.zeros(len(close))
    
    # Calculate intermediate value
    intermediate = np.zeros(len(close))
    mask_valid = (normalized > 0) & (normalized < 1)
    intermediate[mask_valid] = np.log((1.0 - normalized[mask_valid]) / normalized[mask_valid])
    
    # Smooth with EMA
    intermediate_s = pd.Series(intermediate)
    intermediate_smooth = intermediate_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    
    # Final Fisher value
    fisher = 0.66 * intermediate_smooth + 0.67 * np.roll(fisher, 1)
    fisher[0] = intermediate_smooth[0]
    
    # Trigger line (signal line)
    trigger = np.roll(fisher, 1)
    trigger[0] = fisher[0]
    
    return fisher, trigger

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/choppy
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Ensure only one DM is non-zero per bar
    mask_both = (plus_dm > 0) & (minus_dm > 0)
    mask_plus = plus_dm > minus_dm
    plus_dm[mask_both & ~mask_plus] = 0
    minus_dm[mask_both & mask_plus] = 0
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100.0 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100.0 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for macro trend
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h HMA for intermediate trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss and hold logic
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_fisher = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(adx_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === VOLUME FILTER (relaxed) ===
        volume_ok = volume[i] > 0.6 * vol_avg[i]
        
        # === HTF TREND BIAS ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH ===
        trend_strong = adx_14[i] > 18.0  # Avoid dead chop
        
        # === FISHER SIGNALS ===
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5  # Cross above -1.5
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5   # Cross below +1.5
        
        # Fisher reversal (exit signal)
        fisher_reverse_long = fisher[i] < 0.5 and fisher_trigger[i] >= 0.5
        fisher_reverse_short = fisher[i] > -0.5 and fisher_trigger[i] <= -0.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 12h bullish + 4h bullish + Fisher trigger + volume + ADX ---
        if price_above_hma_12h and price_above_hma_4h and fisher_long and volume_ok and trend_strong:
            new_signal = POSITION_SIZE_BASE
            # Add size if Fisher is very oversold (stronger signal)
            if fisher[i] < -1.0:
                new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: 12h bearish + 4h bearish + Fisher trigger + volume + ADX ---
        if price_below_hma_12h and price_below_hma_4h and fisher_short and volume_ok and trend_strong:
            new_signal = -POSITION_SIZE_BASE
            # Add size if Fisher is very overbought (stronger signal)
            if fisher[i] > 1.0:
                new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and no reversal signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if Fisher hasn't reversed
                if not fisher_reverse_long and fisher[i] > -0.5:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if Fisher hasn't reversed
                if not fisher_reverse_short and fisher[i] < 0.5:
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
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher[i]
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_fisher = 0.0
        
        signals[i] = new_signal
    
    return signals