#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d regime filter
# - Primary: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low)
# - HTF: 1d trend regime via ADX > 25 and +DI > -DI (bullish) or -DI > +DI (bearish)
# - Long: Bull Power > 0 AND Bear Power < 0 AND 1d bullish regime (ADX>25 and +DI>-DI)
# - Short: Bear Power > 0 AND Bull Power < 0 AND 1d bearish regime (ADX>25 and -DI>+DI)
# - Exit: Elder Power crosses zero (Bull Power <= 0 for long exit, Bear Power <= 0 for short exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Elder Ray measures power behind moves, 1d ADX filters ranging markets
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits

name = "6h_1d_elderray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for regime
        return np.zeros(n)
    
    # Pre-compute 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components (6h)
    bull_power_6h = high_6h - ema13_6h  # High - EMA13
    bear_power_6h = ema13_6h - low_6h   # EMA13 - Low
    
    # Calculate 1d ADX and DI for regime filter
    # True Range
    tr1 = np.abs(np.roll(high_1d, 1) - np.roll(low_1d, 1))
    tr2 = np.abs(np.roll(high_1d, 1) - np.roll(close_1d, 1))
    tr3 = np.abs(np.roll(low_1d, 1) - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.roll(high_1d, 1) - high_1d
    down_move = low_1d - np.roll(low_1d, 1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    period = 14
    alpha = 1.0 / period
    
    atr_1d = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr_1d
    
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=alpha, adjust=False).mean().values
    
    # Align all HTF indicators to 6h timeframe
    bull_power_6h_aligned = align_htf_to_ltf(prices, df_1d, bull_power_6h)  # Actually 6h, but keeping consistent
    bear_power_6h_aligned = align_htf_to_ltf(prices, df_1d, bear_power_6h)  # Actually 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    # For Elder Ray, we need the actual 6h values (no HTF alignment needed for same timeframe)
    # But we'll use the prices index directly since it's 6h data
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(plus_di_1d_aligned[i]) or
            np.isnan(minus_di_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # 1d regime conditions
        bullish_regime = (adx_1d_aligned[i] > 25.0) and (plus_di_1d_aligned[i] > minus_di_1d_aligned[i])
        bearish_regime = (adx_1d_aligned[i] > 25.0) and (minus_di_1d_aligned[i] > plus_di_1d_aligned[i])
        
        # Elder Ray signals (6h)
        long_signal = bull_power_6h[i] > 0 and bear_power_6h[i] < 0
        short_signal = bear_power_6h[i] > 0 and bull_power_6h[i] < 0
        
        # Exit conditions: Elder Power crosses zero
        exit_long = bull_power_6h[i] <= 0
        exit_short = bear_power_6h[i] <= 0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND 1d bullish regime
            if long_signal and bullish_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power > 0 AND Bull Power < 0 AND 1d bearish regime
            elif short_signal and bearish_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Elder Power crosses zero
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals