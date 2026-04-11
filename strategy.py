#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter
# - Bull Power = Close - EMA13, Bear Power = EMA13 - Low
# - Regime: 1d ADX(14) > 25 for trending, < 20 for ranging
# - In trending regime (ADX > 25): trend follow - long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# - In ranging regime (ADX < 20): mean revert - long when Bull Power < -0.5*ATR and turning up, short when Bear Power < -0.5*ATR and turning down
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Elder Ray measures bull/bear power relative to EMA13, effective in both trending and ranging markets
# - 1d ADX regime filter ensures we apply the right logic for market conditions

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h ATR(14) for regime thresholds
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 6h Elder Ray components
    bull_power = close - ema13  # Close - EMA13
    bear_power = ema13 - low    # EMA13 - Low
    
    # Pre-compute Elder Ray momentum (change from previous bar)
    bull_power_momentum = bull_power - np.roll(bull_power, 1)
    bear_power_momentum = bear_power - np.roll(bear_power, 1)
    bull_power_momentum[0] = 0
    bear_power_momentum[0] = 0
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(atr_14[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(bull_power_momentum[i]) or
            np.isnan(bear_power_momentum[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        
        # Elder Ray components
        bp = bull_power[i]
        br = bear_power[i]
        bpm = bull_power_momentum[i]
        brm = bear_power_momentum[i]
        
        # Regime filters
        adx_value = adx_aligned[i]
        is_trending = adx_value > 25
        is_ranging = adx_value < 20
        
        # Thresholds
        atr_value = atr_14[i]
        bull_threshold = -0.5 * atr_value
        bear_threshold = -0.5 * atr_value
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        if is_trending:
            # Trending regime: trend follow
            # Long when Bull Power > 0 and rising
            if bp > 0 and bpm > 0:
                enter_long = True
            # Short when Bear Power > 0 and rising
            if br > 0 and brm > 0:
                enter_short = True
        elif is_ranging:
            # Ranging regime: mean revert
            # Long when Bull Power < -0.5*ATR and turning up
            if bp < bull_threshold and bpm > 0:
                enter_long = True
            # Short when Bear Power < -0.5*ATR and turning down
            if br < bear_threshold and brm < 0:
                enter_short = True
        # In transition regime (20 <= ADX <= 25), no new entries
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if power deteriorates or opposite signal
            exit_long = (bp <= 0) or (br > 0 and brm > 0)
        elif position == -1:
            # Exit short if power deteriorates or opposite signal
            exit_short = (br <= 0) or (bp > 0 and bpm > 0)
        
        # Track entry price for reference (not used in stops, but kept for consistency)
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals