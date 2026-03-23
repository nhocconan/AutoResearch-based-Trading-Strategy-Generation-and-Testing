#!/usr/bin/env python3
"""
Experiment #049: 4h Primary + 1d HTF — KAMA + Fisher Transform + Donchian Breakout

Hypothesis: Current best uses CRSI + Choppiness + Vol Spike. I'm testing a DIFFERENT 
combination that should work better in bear/range markets (2025 test period).

Key innovations:
1. KAMA (Kaufman Adaptive Moving Average): Adapts to volatility, less whipsaw than HMA/EMA
   - ER (Efficiency Ratio) determines smoothing constant
   - Works better in choppy BTC/ETH markets than fixed-period MAs
2. FISHER TRANSFORM: Superior reversal detection vs RSI (Ehlers research)
   - Long when Fisher crosses above -1.5 from below
   - Short when Fisher crosses below +1.5 from above
   - Catches bear market rally reversals better than RSI
3. DONCHIAN BREAKOUT CONFIRMATION: Adds momentum filter to avoid fakeouts
   - Long: price > Donchian(20) high + KAMA bullish
   - Short: price < Donchian(20) low + KAMA bearish
4. CHOPPINESS REGIME: Switch between mean-revert (chop>55) and trend-follow (chop<45)
5. ASYMMETRIC SIZING: 0.35 for high-confidence trend, 0.25 for mean-revert

Why this should beat current best (Sharpe=0.424):
- KAMA reduces whipsaw in 2022 crash vs HMA
- Fisher Transform catches reversals earlier than CRSI
- Donchian confirmation filters false breakouts
- Still generates 30-60 trades/year (4h timeframe target)

Entry conditions (LOOSE enough for trades):
- Mean-revert long: Fisher < -1.5 + chop > 55 + price > 1d KAMA
- Mean-revert short: Fisher > +1.5 + chop > 55 + price < 1d KAMA
- Trend long: Fisher cross up + chop < 45 + price > Donchian(20) high + 1d KAMA bullish
- Trend short: Fisher cross down + chop < 45 + price < Donchian(20) low + 1d KAMA bearish

Position size: 0.25-0.35 (discrete, within limits)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_donchian_chop_regime_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    KAMA adapts smoothing based on market efficiency:
    - High ER (trending) = fast smoothing (reacts quickly)
    - Low ER (choppy) = slow smoothing (filters noise)
    
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio calculation
    price_change = np.abs(close_s.diff(er_period).values)
    volatility = pd.Series(np.abs(close_s.diff().values)).rolling(window=er_period, min_periods=er_period).sum().values
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = volatility > 1e-10
    er[mask] = price_change[mask] / volatility[mask]
    er[~mask] = 0.0
    er[:er_period] = 0.0  # Not enough data at start
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize with price
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Fisher Transform normalizes price to Gaussian distribution,
    making reversals easier to spot.
    
    Steps:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    """
    n = len(high)
    typical_price = (high + low) / 2.0
    
    # Normalize price to -1 to +1
    fisher_input = np.zeros(n)
    for i in range(period, n):
        highest = np.max(typical_price[i-period+1:i+1])
        lowest = np.min(typical_price[i-period+1:i+1])
        price_range = highest - lowest
        if price_range > 1e-10:
            fisher_input[i] = 2.0 * (typical_price[i] - lowest) / price_range - 1.0
        else:
            fisher_input[i] = 0.0
    
    # Clamp to avoid ln(0) or ln(inf)
    fisher_input = np.clip(fisher_input, -0.999, 0.999)
    
    # Fisher transform
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        fisher[i] = 0.5 * np.log((1.0 + fisher_input[i]) / (1.0 - fisher_input[i] + 1e-10))
        if i > period:
            fisher_signal[i] = fisher[i-1]  # Previous bar's fisher for crossover
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    return donchian_high, donchian_low, donchian_mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for macro trend bias
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.35  # Higher confidence trend trades
    SIZE_MR = 0.25     # Lower confidence mean-revert trades
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(donchian_high[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0
        is_trending = chop_value < 45.0
        
        # === 1D MACRO BIAS ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # 1d KAMA slope (3-bar lookback)
        kama_1d_slope_bull = False
        kama_1d_slope_bear = False
        if i >= 3 and not np.isnan(kama_1d_aligned[i-3]):
            kama_1d_slope_bull = kama_1d_aligned[i] > kama_1d_aligned[i-3]
            kama_1d_slope_bear = kama_1d_aligned[i] < kama_1d_aligned[i-3]
        
        # === 4H KAMA TREND ===
        kama_4h_slope_bull = False
        kama_4h_slope_bear = False
        if i >= 3:
            kama_4h_slope_bull = kama_4h[i] > kama_4h[i-3]
            kama_4h_slope_bear = kama_4h[i] < kama_4h[i-3]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher crossover signals
        fisher_cross_up = False
        fisher_cross_down = False
        if i >= 2 and not np.isnan(fisher_signal[i]):
            fisher_cross_up = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
            fisher_cross_down = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_high = close[i] > donchian_high[i-1] if i >= 1 else False
        donchian_breakout_low = close[i] < donchian_low[i-1] if i >= 1 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion ---
        if is_ranging:
            # Long: Fisher oversold + price above 1d KAMA (macro support)
            if fisher_oversold and price_above_kama_1d:
                new_signal = SIZE_MR
            
            # Short: Fisher overbought + price below 1d KAMA (macro resistance)
            elif fisher_overbought and price_below_kama_1d:
                new_signal = -SIZE_MR
        
        # --- TRENDING REGIME: Trend Following ---
        elif is_trending:
            # Long: Fisher cross up + Donchian breakout + KAMA bullish + 1d bullish
            if fisher_cross_up and donchian_breakout_high:
                if kama_4h_slope_bull and price_above_kama_1d:
                    new_signal = SIZE_TREND
                elif kama_4h_slope_bull:  # Weaker signal without 1d confirmation
                    new_signal = SIZE_MR
            
            # Short: Fisher cross down + Donchian breakdown + KAMA bearish + 1d bearish
            elif fisher_cross_down and donchian_breakout_low:
                if kama_4h_slope_bear and price_below_kama_1d:
                    new_signal = -SIZE_TREND
                elif kama_4h_slope_bear:  # Weaker signal without 1d confirmation
                    new_signal = -SIZE_MR
        
        # === HOLD POSITION LOGIC ===
        # Keep position if no new signal and not stopped out
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
        # Exit long if regime changes to strongly trending bearish
        if in_position and position_side > 0:
            if is_trending and kama_4h_slope_bear and price_below_kama_1d:
                new_signal = 0.0
        
        # Exit short if regime changes to strongly trending bullish
        if in_position and position_side < 0:
            if is_trending and kama_4h_slope_bull and price_above_kama_1d:
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
                # Position flip
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