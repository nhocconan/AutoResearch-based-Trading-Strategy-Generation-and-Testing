#!/usr/bin/env python3
"""
Experiment #036: 12h Primary + 1d HTF — Fisher Transform + KAMA Adaptive Trend + Choppiness Regime

Hypothesis: Based on research showing Ehlers Fisher Transform catches reversals in bear markets
better than RSI, combined with KAMA (Kaufman Adaptive Moving Average) which adapts to market
efficiency ratio. This should work better than HMA in choppy conditions.

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, crosses -1.5 for long, +1.5 for short. Superior reversal detection.
2. KAMA (Kaufman Adaptive): ER-based smoothing that adapts to trending vs ranging markets.
3. CHOPPINESS INDEX: Regime filter (CHOP>55 = range, CHOP<45 = trend).
4. DONCHIAN BREAKOUT: 20-period high/low for trend confirmation.
5. 1d KAMA for macro bias (slower, more reliable than HMA).

Why 12h works:
- Targets 20-50 trades/year (fee-efficient per Rule 10)
- Less noise than 4h, more signals than 1d
- Proven in exp #026 (Sharpe=0.354) and #032 (Sharpe=0.419)

Entry conditions (LOOSE enough to generate trades on ALL symbols):
- Long range: Fisher < -1.2 + CHOP > 50 + price > 1d KAMA + near BB lower
- Short range: Fisher > +1.2 + CHOP > 50 + price < 1d KAMA + near BB upper
- Long trend: Fisher crosses up + KAMA bullish + Donchian breakout + ADX > 20
- Short trend: Fisher crosses down + KAMA bearish + Donchian breakdown + ADX > 20

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_chop_donchian_1d_v1"
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

def calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/31):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on Efficiency Ratio (ER).
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    signal = np.abs(close_s - close_s.shift(er_period))
    noise = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    er = signal / (noise + 1e-10)
    er = er.fillna(0)
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    X = 0.33 * 2 * ((close - LL) / (HH - LL) - 0.5) + 0.67 * X_prev
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh == ll:
            continue
        
        # Normalize price position within range
        x = 0.33 * 2 * ((high[i] + low[i]) / 2 - ll) / (hh - ll) - 0.33
        
        # Apply smoothing
        if i > period:
            x = 0.33 * 2 * ((high[i] + low[i]) / 2 - ll) / (hh - ll) - 0.33
            x_prev = 0.33 * 2 * ((high[i-1] + low[i-1]) / 2 - 
                   np.min(low[i-period:i]) / (np.max(high[i-period:i]) - np.min(low[i-period:i]) + 1e-10)) - 0.33
            x = 0.2 * x + 0.8 * x_prev
        
        # Clamp to avoid log errors
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        if i > 0:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_dm[(high_s.diff() <= -low_s.diff())] = 0
    minus_dm[(-low_s.diff() <= high_s.diff())] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for macro bias
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    kama_12h = calculate_kama(close, er_period=10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(atr_7[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(bb_upper[i]) or np.isnan(kama_12h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === VOLATILITY FILTER ===
        vol_spike = (atr_7[i] / atr_14[i]) > 1.5  # High vol = reversion opportunity
        vol_normal = (atr_7[i] / atr_14[i]) <= 1.3  # Normal vol = trend following OK
        
        # === 1D MACRO BIAS ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 12H TREND BIAS ===
        kama_12h_slope_bull = kama_12h[i] > kama_12h[i-5] if i >= 5 else False
        kama_12h_slope_bear = kama_12h[i] < kama_12h[i-5] if i >= 5 else False
        price_above_kama_12h = close[i] > kama_12h[i]
        price_below_kama_12h = close[i] < kama_12h[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Lower threshold for more range trades
        is_trending = chop_value < 45.0 and adx[i] > 20.0  # Require ADX confirmation
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.2  # Looser than -1.5 for more trades
        fisher_overbought = fisher[i] > 1.2  # Looser than +1.5 for more trades
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1] if i > 0 else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakdown_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with Fisher ---
        if is_ranging:
            # Long: Fisher oversold + near BB lower + 1d bias OR vol spike
            if fisher_oversold and (price_near_bb_lower or vol_spike):
                if price_above_kama_1d or vol_spike:  # Easier entry with vol spike
                    new_signal = POSITION_SIZE
            
            # Short: Fisher overbought + near BB upper + 1d bias OR vol spike
            elif fisher_overbought and (price_near_bb_upper or vol_spike):
                if price_below_kama_1d or vol_spike:  # Easier entry with vol spike
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with Donchian ---
        elif is_trending:
            # Long: Fisher cross up + KAMA bullish + Donchian breakout + ADX confirmation
            if (fisher_cross_up or fisher_oversold) and kama_12h_slope_bull and price_above_kama_12h:
                if donchian_breakout_up or price_above_kama_1d:  # Either breakout or 1d confirmation
                    new_signal = POSITION_SIZE
            
            # Short: Fisher cross down + KAMA bearish + Donchian breakdown + ADX confirmation
            elif (fisher_cross_down or fisher_overbought) and kama_12h_slope_bear and price_below_kama_12h:
                if donchian_breakdown_down or price_below_kama_1d:  # Either breakdown or 1d confirmation
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime changes from ranging to strongly trending bearish
        if in_position and position_side > 0:
            if is_trending and kama_12h_slope_bear and price_below_kama_1d:
                new_signal = 0.0
        
        # Exit short if regime changes from ranging to strongly trending bullish
        if in_position and position_side < 0:
            if is_trending and kama_12h_slope_bull and price_above_kama_1d:
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