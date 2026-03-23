#!/usr/bin/env python3
"""
Experiment #686: 12h Primary + 1d HTF — Vol Spike Mean Reversion + Regime Filter

Hypothesis: After 12h Choppiness+CRSI strategies failed (#676, #682, #683 all negative Sharpe),
I'm switching to a COMPLETELY DIFFERENT approach based on proven literature:

1. Volatility Spike Detection: ATR(7)/ATR(30) ratio identifies panic/extreme vol events
2. Mean Reversion Entry: Price outside Bollinger(20, 2.0) during vol spike = fade opportunity
3. Regime Filter: BB Width percentile distinguishes trend vs range (avoid fading strong trends)
4. 1d HMA Bias: Only take mean-reversion trades aligned with higher timeframe direction
5. Exit: When vol normalizes (ATR ratio < 1.2) or stoploss hit

Why this differs from ALL failed 12h attempts:
- NO Choppiness Index (failed 3+ times on 12h)
- NO Connors RSI (failed 3+ times on 12h)
- Uses VOLATILITY signal instead of momentum/oscillator
- Mean-reversion edge documented in quantitative literature (Vol Spike Reversion)
- Specifically targets "panic capitulation" longs and "euphoria blowoff" shorts

Position sizing: 0.30 discrete (Rule 4)
Target: 30-50 trades/year on 12h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_volspike_bb_regime_1d_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_bb_width(high, low, close, period=20):
    """Calculate Bollinger Band Width (normalized)."""
    upper, lower, sma = calculate_bollinger(close, period, 2.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_width = (upper - lower) / (sma + 1e-10)
    return bb_width

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, 20, 2.0)
    bb_width = calculate_bb_width(high, low, close, 20)
    
    # Calculate BB Width percentile for regime detection (rolling 100 bars)
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=100).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    ).values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width_pct[i]):
            continue
        if atr_30[i] == 0 or atr_7[i] == 0:
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 1.5  # 50% above normal vol
        vol_normal = atr_ratio < 1.2  # Vol normalized
        
        # === BOLLINGER POSITION ===
        price_below_bb = close[i] < bb_lower[i]  # Below lower band
        price_above_bb = close[i] > bb_upper[i]  # Above upper band
        
        # === REGIME DETECTION (BB Width Percentile) ===
        is_range_regime = bb_width_pct[i] < 0.5  # Lower half = range/compression
        is_trend_regime = bb_width_pct[i] > 0.7  # Upper 30% = expansion/trend
        
        # === 1D TREND BIAS (HMA slope over 5 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Vol spike + price below BB + 1d bias not strongly bear ---
        # Fade panic selling when vol spikes and price extends below BB
        if vol_spike and price_below_bb:
            # Only enter if 1d trend is not strongly bearish (avoid catching falling knife in strong downtrend)
            if not (hma_1d_slope_bear and price_below_hma_1d):
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: Vol spike + price above BB + 1d bias not strongly bull ---
        # Fade euphoria buying when vol spikes and price extends above BB
        elif vol_spike and price_above_bb:
            # Only enter if 1d trend is not strongly bullish
            if not (hma_1d_slope_bull and price_above_hma_1d):
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === EXIT ON VOL NORMALIZATION ===
        if in_position and vol_normal:
            new_signal = 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_30[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_30[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON 1D TREND FLIP (against position) ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
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