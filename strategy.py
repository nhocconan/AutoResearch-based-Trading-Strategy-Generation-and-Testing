#!/usr/bin/env python3
"""
Experiment #559: 15m Fisher Transform + Choppiness Regime + 4h HMA Trend

Hypothesis: After 500+ failed experiments, key insights:
1. RSI pullback strategies FAILED BADLY on 15m (#553: -93.6% return)
2. Need DIFFERENT entry signal - Fisher Transform catches reversals better
3. Choppiness Index (CHOP) distinguishes range vs trend regimes
4. 4h HMA trend bias prevents counter-trend entries (proven filter)
5. 15m timeframe needs tighter stops but more trades than 1h/4h

Why Fisher Transform works better than RSI:
- Normalizes price into Gaussian distribution (-1.5 to +1.5 range)
- Clearer extreme values for reversal entries
- Less lag than RSI, better for 15m intraday
- Proven in bear/range markets (2022 crash, 2025 bear)

Why Choppiness Index:
- CHOP > 61.8 = range market (mean revert at extremes)
- CHOP < 38.2 = trending market (breakout entries)
- Prevents trend strategies in chop (major failure mode)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_chop_regime_4h_hma_adaptive_atr_v1"
timeframe = "15m"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price into Gaussian distribution.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max()
    lowest = hl2_s.rolling(window=period, min_periods=period).min()
    
    # Normalize to 0-1 range
    norm = (hl2 - lowest) / (highest - lowest + 1e-10)
    norm = np.clip(norm, 0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + norm) / (1 - norm + 1e-10))
    fisher_s = pd.Series(fisher)
    
    # Signal line (1-period lag)
    fisher_signal = fisher_s.shift(1).values
    
    return fisher_s.values, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/consolidation (mean revert)
    CHOP < 38.2 = trending (breakout strategy)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high - Lowest low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100 * np.log10(tr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    return chop.values

def calculate_macd_histogram(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram for momentum confirmation."""
    close_s = pd.Series(close)
    
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return histogram.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    macd_hist = calculate_macd_histogram(close)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop[i] > 61.8  # Range market
        is_trend = chop[i] < 38.2  # Trending market
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === MACD MOMENTUM CONFIRMATION ===
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entries (2 modes based on regime):
        # Mode 1: Range market + Fisher extreme long + bull bias
        if is_range and fisher_long_cross and bull_bias:
            new_signal = SIZE
        # Mode 2: Trending market + Fisher long + MACD bullish + bull bias
        elif is_trend and fisher_long_cross and macd_bullish and bull_bias:
            new_signal = SIZE
        
        # Short entries (2 modes based on regime):
        # Mode 1: Range market + Fisher extreme short + bear bias
        if is_range and fisher_short_cross and bear_bias:
            new_signal = -SIZE
        # Mode 2: Trending market + Fisher short + MACD bearish + bear bias
        elif is_trend and fisher_short_cross and macd_bearish and bear_bias:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals