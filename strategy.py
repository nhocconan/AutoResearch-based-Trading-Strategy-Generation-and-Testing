#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX) and volume confirmation
# - Bull Power = High - EMA(13); Bear Power = EMA(13) - Low
# - Regime: Trending (1d ADX > 25) vs Ranging (1d ADX < 20) with hysteresis
# - In Trending: Trade breakouts in direction of 13-period EMA (momentum)
# - In Ranging: Fade extremes when Bull/Bear Power diverges from price (mean reversion)
# - Volume confirmation: Current volume > 1.3x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Elder Ray captures both trend and reversal opportunities with clear rules

name = "6h_1d_elder_ray_regime_v4"
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
    
    # Load 1d data ONCE before loop for regime filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d ADX(14) for regime detection
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
    
    # Pre-compute 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 6h ATR for stoploss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 6h price position for divergence detection
    price_above_ema = close > ema_13
    price_below_ema = close < ema_13
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_14[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Regime detection with hysteresis
        adx_val = adx_aligned[i]
        if adx_val > 25:
            regime = 'trending'  # Strong trend
        elif adx_val < 20:
            regime = 'ranging'   # Weak trend/ranging
        else:
            regime = regime      # Hold previous regime (hysteresis)
        
        # Initialize regime on first valid bar
        if i == 100:
            regime = 'trending' if adx_val > 22.5 else 'ranging'
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        if regime == 'trending':
            # In trending markets: trade breakouts in direction of momentum
            # Long: price above EMA and Bull Power increasing (strong buying pressure)
            # Short: price below EMA and Bear Power increasing (strong selling pressure)
            if i >= 1:
                bull_power_rising = bull_power[i] > bull_power[i-1]
                bear_power_rising = bear_power[i] > bear_power[i-1]
                
                enter_long = (close_price > ema_13[i]) and bull_power_rising and vol_confirm
                enter_short = (close_price < ema_13[i]) and bear_power_rising and vol_confirm
        
        else:  # regime == 'ranging'
            # In ranging markets: fade extremes when Elder Power diverges from price
            # Long: price below EMA but Bull Power rising (bullish divergence)
            # Short: price above EMA but Bear Power rising (bearish divergence)
            if i >= 1:
                bull_power_rising = bull_power[i] > bull_power[i-1]
                bear_power_rising = bear_power[i] > bear_power[i-1]
                
                enter_long = (close_price < ema_13[i]) and bull_power_rising and vol_confirm
                enter_short = (close_price > ema_13[i]) and bear_power_rising and vol_confirm
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: momentum weakening or opposite divergence
            if i >= 1:
                bull_power_falling = bull_power[i] < bull_power[i-1]
                bear_power_rising = bear_power[i] > bear_power[i-1]
                price_below_ema = close_price < ema_13[i]
                
                exit_long = bull_power_falling or (bear_power_rising and price_below_ema) or (close_price <= ema_13[i] - 1.5 * atr_14[i])
        elif position == -1:
            # Exit short: momentum weakening or opposite divergence
            if i >= 1:
                bear_power_falling = bear_power[i] < bear_power[i-1]
                bull_power_rising = bull_power[i] > bull_power[i-1]
                price_above_ema = close_price > ema_13[i]
                
                exit_short = bear_power_falling or (bull_power_rising and price_above_ema) or (close_price >= ema_13[i] + 1.5 * atr_14[i])
        
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