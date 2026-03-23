#!/usr/bin/env python3
"""
Experiment #033: 1d Primary + 1w HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Daily timeframe with Ehlers Fisher Transform for entry timing + KAMA for 
adaptive trend following will outperform RSI-based strategies in bear/range markets. 
Key insight: Fisher Transform normalizes price to Gaussian distribution, catching 
reversals better than RSI during 2022 crash and 2025 bear market. KAMA adapts to 
volatility regime automatically (fast in trends, slow in chop).

Strategy Logic:
1. FISHER TRANSFORM (period=9): Entry trigger at extremes (-1.5 long, +1.5 short)
2. KAMA TREND (ER=10, fast=2/30, slow=2/30): Adaptive moving average for trend direction
3. CHOPPINESS INDEX (14): Regime filter - only trend follow when CHOP<45
4. 1w HMA: Macro bias filter (trade with weekly direction)
5. VOLUME CONFIRMATION: Breakouts need 1.5x avg volume to reduce false signals
6. ATR(14) trailing stoploss: 2.5*ATR with asymmetric sizing (0.35 long, 0.25 short)

Why this should beat #027 (Sharpe=0.395):
- Fisher Transform catches reversals better than RSI in bear markets (2022, 2025)
- KAMA adapts to volatility - less whipsaw than fixed HMA
- Volume filter reduces false Donchian breakouts
- Asymmetric sizing matches crypto's long-term upward bias

Position size: 0.35 long, 0.25 short (discrete, within 0.20-0.40 range)
Stoploss: 2.5*ATR trailing
Target: 25-40 trades/year on 1d timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_kama_chop_volume_1w_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.nan
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    
    er = change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc[i]) or i < period:
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal detection.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price to -1 to +1 range
        range_val = hh - ll
        if range_val < 1e-10:
            continue
        
        x = (2.0 * (close[i] - ll) / range_val) - 1.0
        x = np.clip(x, -0.999, 0.999)  # Prevent log domain error
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Trigger line (previous fisher value)
        if i > 0:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average for confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, asymmetric for long bias)
    POSITION_SIZE_LONG = 0.35
    POSITION_SIZE_SHORT = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(kama_10[i]):
            continue
        if atr_14[i] == 0 or np.isnan(vol_sma_20[i]):
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_trending = chop_value < 45.0  # Trend market
        is_ranging = chop_value > 55.0  # Range market
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1] if i > 0 else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1] if i > 0 else False
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        kama_slope_up = kama_10[i] > kama_10[i-5] if i > 5 else False
        kama_slope_down = kama_10[i] < kama_10[i-5] if i > 5 else False
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_sma_20[i] if vol_sma_20[i] > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Fisher + KAMA trend follow ---
        if is_trending:
            # Long: Fisher oversold/cross up + KAMA bullish + weekly confirms
            if (fisher_oversold or fisher_cross_up) and kama_bullish:
                if price_above_hma_1w or kama_slope_up:
                    new_signal = POSITION_SIZE_LONG
            
            # Short: Fisher overbought/cross down + KAMA bearish + weekly confirms
            elif (fisher_overbought or fisher_cross_down) and kama_bearish:
                if price_below_hma_1w or kama_slope_down:
                    new_signal = -POSITION_SIZE_SHORT
        
        # --- RANGING REGIME: Mean reversion at Fisher extremes ---
        elif is_ranging:
            # Long: Fisher deeply oversold + price above weekly HMA
            if fisher[i] < -2.0 and price_above_hma_1w:
                new_signal = POSITION_SIZE_LONG
            
            # Short: Fisher deeply overbought + price below weekly HMA
            elif fisher[i] > 2.0 and price_below_hma_1w:
                new_signal = -POSITION_SIZE_SHORT
        
        # --- FALLBACK: KAMA crossover with volume ---
        if new_signal == 0.0:
            # Long: Price crosses above KAMA + volume spike + weekly helps
            if close[i] > kama_10[i] and close[i-1] <= kama_10[i-1]:
                if volume_confirmed and (price_above_hma_1w or fisher[i] > -1.0):
                    new_signal = POSITION_SIZE_LONG
            
            # Short: Price crosses below KAMA + volume spike + weekly helps
            elif close[i] < kama_10[i] and close[i-1] >= kama_10[i-1]:
                if volume_confirmed and (price_below_hma_1w or fisher[i] < 1.0):
                    new_signal = -POSITION_SIZE_SHORT
        
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
        
        # === EXIT ON REGIME/TREND CHANGE ===
        # Exit long if weekly trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1w and kama_bearish and chop_value < 40:
                new_signal = 0.0
        
        # Exit short if weekly trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1w and kama_bullish and chop_value < 40:
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