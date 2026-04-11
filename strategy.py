#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# - Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# - Regime filter: 1d ADX(14) > 20 for trending, < 20 for ranging
# - In trending regime (ADX > 20): Long when Bull Power > 0 and rising, Short when Bear Power > 0 and rising
# - In ranging regime (ADX <= 20): Long when Bull Power < -0.5*ATR and turning up, Short when Bear Power < -0.5*ATR and turning up
# - Volume confirmation: current volume > 1.3x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Elder Ray measures bull/bear strength relative to EMA; works in both bull and bear markets
# - 1d ADX regime filter adapts to market conditions, reducing whipsaw
# - Volume confirmation ensures breakouts have participation

name = "6h_1d_elder_ray_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for regime filter (ADX) and volume
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
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h ATR for stoploss and regime thresholds
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 6h Bull Power and Bear Power (Elder Ray)
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Pre-compute rising/falling power (1-period change)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    bull_power_turning_up = (bull_power > np.roll(bull_power, 1)) & (np.roll(bull_power, 1) <= np.roll(bull_power, 2))
    bear_power_turning_up = (bear_power > np.roll(bear_power, 1)) & (np.roll(bear_power, 1) <= np.roll(bear_power, 2))
    
    # Handle first values
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    bull_power_turning_up[0] = False
    bear_power_turning_up[0] = False
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Elder Ray values
        bp = bull_power[i]
        br = bear_power[i]
        
        # Power momentum
        bp_rising = bull_power_rising[i]
        br_rising = bear_power_rising[i]
        bp_turning_up = bull_power_turning_up[i]
        br_turning_up = bear_power_turning_up[i]
        
        # Regime filter: 1d ADX > 20 = trending, <= 20 = ranging
        adx_value = adx_aligned[i]
        is_trending = adx_value > 20
        is_ranging = adx_value <= 20
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20_aligned[i]
        
        # ATR-based thresholds for ranging regime
        atr_value = atr_14[i]
        bull_oversold = bp < -0.5 * atr_value
        bear_oversold = br < -0.5 * atr_value
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        if is_trending:
            # Trending regime: follow momentum
            enter_long = (bp > 0 and bp_rising and vol_confirm)
            enter_short = (br > 0 and br_rising and vol_confirm)
        else:  # ranging regime
            # Ranging regime: mean reversion from extremes
            enter_long = (bull_oversold and bp_turning_up and vol_confirm)
            enter_short = (bear_oversold and br_turning_up and vol_confirm)
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if power deteriorates or ATR stop
            exit_long = (bp < 0) or (close_price <= entry_price - 2.0 * atr_value)
        elif position == -1:
            # Exit short if power deteriorates or ATR stop
            exit_short = (br < 0) or (close_price >= entry_price + 2.0 * atr_value)
        
        # Track entry price for stoploss calculation
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