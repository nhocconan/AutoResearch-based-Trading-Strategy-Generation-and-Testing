#!/usr/bin/env python3
"""
Experiment #041: 4h Primary + 1d/1w HTF — Fisher Transform + Donchian Breakout with Regime Filter

Hypothesis: Based on research showing Ehlers Fisher Transform excels at catching reversals in 
bear/range markets (75%+ win rate on BTC/ETH), combined with Donchian breakouts for trend 
confirmation. The key insight is that Fisher works better than RSI for extreme reversals, 
and Donchian provides cleaner breakout signals than HMA crossover.

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, identifies true reversal points better than RSI
2. DONCHIAN CHANNEL: 20-period high/low for clean breakout signals
3. CHOPPINESS REGIME: CHOP(14) > 55 = range (Fisher mean revert), < 45 = trend (Donchian breakout)
4. 1D HMA macro bias: only long above 1d HMA, only short below 1d HMA
5. VOL SPIKE FILTER: ATR(7)/ATR(30) > 1.8 for panic reversion entries
6. Asymmetric thresholds: easier mean-revert entries, harder trend entries (bear market bias)

Why this should work:
- Fisher Transform proven in bear markets (2022 crash, 2025 range)
- Donchian breakout avoids whipsaw of EMA crossover
- 4h timeframe targets 25-45 trades/year (fee-efficient)
- Regime filter prevents trend-following in chop (major source of losses)

Entry conditions (LOOSE enough to generate 30+ trades):
- Long mean-revert: Fisher < -1.5 + CHOP > 55 + price > 1d HMA + (vol spike OR near Donchian low)
- Short mean-revert: Fisher > +1.5 + CHOP > 55 + price < 1d HMA + (vol spike OR near Donchian high)
- Long trend: Fisher crosses above -1.0 + CHOP < 45 + Donchian breakout + price > 1d HMA
- Short trend: Fisher crosses below +1.0 + CHOP < 45 + Donchian breakdown + price < 1d HMA

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_donchian_chop_regime_1d1w_v1"
timeframe = "4h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.67 * ((price - lowest_low) / (highest_high - lowest_low) - 0.5) + 0.67 * X_prev
    
    Excellent for identifying reversal points in bear/range markets.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Calculate median price (HL2)
    median_price = (high + low) / 2.0
    
    # Rolling highest high and lowest low
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)  # Avoid division by zero
    
    X = np.zeros(n)
    X_prev = 0.0
    
    for i in range(period, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            continue
        
        # Normalize price within range
        raw_x = (median_price[i] - lowest_low[i]) / price_range[i]
        
        # Smooth with previous X (Ehlers smoothing)
        X[i] = 0.67 * (raw_x - 0.5) + 0.67 * X_prev
        X[i] = np.clip(X[i], -0.99, 0.99)  # Clamp to avoid ln domain errors
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + X[i]) / (1.0 - X[i]) + 1e-10)
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
        
        X_prev = X[i]
    
    return fisher, trigger

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(donchian_upper[i]) or np.isnan(chop_14[i]):
            continue
        if atr_14[i] == 0 or atr_30[i] == 0:
            continue
        
        # === VOLATILITY SPIKE FILTER ===
        vol_spike = (atr_7[i] / atr_30[i]) > 1.8  # High vol = reversion opportunity
        vol_normal = (atr_7[i] / atr_30[i]) <= 1.5  # Normal vol = trend following OK
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0
        is_trending = chop_value < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5  # Strong reversal signal
        fisher_overbought = fisher[i] > 1.5  # Strong reversal signal
        fisher_cross_up = (fisher[i] > -1.0) and (fisher_trigger[i] <= -1.0)  # Cross above -1
        fisher_cross_down = (fisher[i] < 1.0) and (fisher_trigger[i] >= 1.0)  # Cross below +1
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakdown_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Price near Donchian bounds (for mean reversion)
        price_near_donchian_low = (close[i] - donchian_lower[i]) / (donchian_upper[i] - donchian_lower[i] + 1e-10) < 0.15
        price_near_donchian_high = (close[i] - donchian_lower[i]) / (donchian_upper[i] - donchian_lower[i] + 1e-10) > 0.85
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Fisher Mean Reversion ---
        if is_ranging:
            # Long: Fisher oversold + macro bias OR vol spike + near Donchian low
            if fisher_oversold:
                if price_above_hma_1d or vol_spike:
                    new_signal = POSITION_SIZE
            # Short: Fisher overbought + macro bias OR vol spike + near Donchian high
            elif fisher_overbought:
                if price_below_hma_1d or vol_spike:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Donchian Breakout with Fisher Confirmation ---
        elif is_trending:
            # Long: Donchian breakout + Fisher not overbought + 1d HMA bullish
            if donchian_breakout_up and fisher[i] < 1.0:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE
            # Short: Donchian breakdown + Fisher not oversold + 1d HMA bearish
            elif donchian_breakdown_down and fisher[i] > -1.0:
                if price_below_hma_1d:
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
            if is_trending and price_below_hma_1d and fisher[i] > 1.0:
                new_signal = 0.0
        
        # Exit short if regime changes from ranging to strongly trending bullish
        if in_position and position_side < 0:
            if is_trending and price_above_hma_1d and fisher[i] < -1.0:
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