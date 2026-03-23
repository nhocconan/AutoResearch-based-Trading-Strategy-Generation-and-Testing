#!/usr/bin/env python3
"""
Experiment #476: 12h Primary + 1d HTF — Fisher Transform + Donchian Breakout + Volume + ADX

Hypothesis: Based on research showing Ehlers Fisher Transform excels at catching reversals in bear/range
markets (2025 test period), combined with Donchian breakouts for trend confirmation. Key innovations:
1. Fisher Transform (period=9) - normalized oscillator, long when crosses above -1.5, short below +1.5
2. Donchian(20) breakout - price breaks 20-bar high/low for trend confirmation
3. ADX(14) > 20 for trend strength filter (prevents chop entries)
4. Volume spike confirmation (vol > 1.5x 20-bar avg) for breakout validity
5. 1d Fisher for HTF bias alignment (simpler than dual HTF)
6. ATR(14) trailing stop at 2.5x for risk management
7. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work: Fisher Transform is proven for bear market reversals (unlike RSI which lags).
Donchian breakouts capture sustained trends. Volume filter reduces false breakouts. ADX prevents
choppy market whipsaws. 12h TF targets 20-50 trades/year (fee-efficient). 1d HTF ensures we trade
with higher timeframe momentum. This is DIFFERENT from failed CRSI+Chop combinations.

Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_donchian_vol_adx_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    # Normalize price to -1 to +1 range
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest - lowest < 1e-10:
            continue
        
        # Normalize to 0-1, then scale to -0.99 to +0.99
        normalized = 2.0 * ((close[i] - lowest) / (highest - lowest)) - 1.0
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Signal line (1-bar lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-bar high/low)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth with Wilder's method
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * (plus_dm_s / (tr_s + 1e-10))
        minus_di = 100.0 * (minus_dm_s / (tr_s + 1e-10))
        
        # DX
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        # ADX (smoothed DX)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes (volume > threshold * average)."""
    n = len(volume)
    spike = np.zeros(n, dtype=bool)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period, n):
        if vol_avg[i] > 1e-10 and volume[i] > threshold * vol_avg[i]:
            spike[i] = True
    
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    fisher_12h, fisher_signal_12h = calculate_fisher(close, period=9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx_14 = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # Calculate and align HTF indicators (1d Fisher for bias)
    fisher_1d_raw, _ = calculate_fisher(df_1d['close'].values, period=9)
    fisher_1d_aligned = align_htf_to_ltf(prices, df_1d, fisher_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(fisher_12h[i]) or np.isnan(fisher_signal_12h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(adx_14[i]):
            continue
        if np.isnan(fisher_1d_aligned[i]):
            continue
        
        # === TREND STRENGTH (ADX) ===
        is_trending = adx_14[i] > 20.0  # Trend market
        is_strong_trend = adx_14[i] > 25.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher_12h[i] > -1.5 and fisher_signal_12h[i] <= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher_12h[i] < 1.5 and fisher_signal_12h[i] >= 1.5
        
        # Fisher extreme levels (stronger signal)
        fisher_oversold = fisher_12h[i] < -1.0
        fisher_overbought = fisher_12h[i] > 1.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === HTF BIAS (1d Fisher) ===
        htf_bullish = fisher_1d_aligned[i] > 0.0
        htf_bearish = fisher_1d_aligned[i] < 0.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES (multiple confluence required)
        long_score = 0
        
        # Fisher reversal signal (required)
        if fisher_long or (fisher_oversold and fisher_12h[i] > fisher_signal_12h[i]):
            long_score += 2
        
        # Donchian breakout confirmation
        if breakout_long:
            long_score += 2
        
        # Volume spike confirmation (important for breakouts)
        if vol_confirmed:
            long_score += 1
        
        # HTF bias alignment
        if htf_bullish:
            long_score += 1
        
        # ADX trend strength (prefer trending for breakouts)
        if is_trending:
            long_score += 1
        
        # Enter long if score >= 4 (need multiple confirmations)
        if long_score >= 4:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # Fisher reversal signal (required)
            if fisher_short or (fisher_overbought and fisher_12h[i] < fisher_signal_12h[i]):
                short_score += 2
            
            # Donchian breakout confirmation
            if breakout_short:
                short_score += 2
            
            # Volume spike confirmation
            if vol_confirmed:
                short_score += 1
            
            # HTF bias alignment
            if htf_bearish:
                short_score += 1
            
            # ADX trend strength
            if is_trending:
                short_score += 1
            
            if short_score >= 4:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and fisher_12h[i] > 0.0 and htf_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and fisher_12h[i] < 0.0 and htf_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals