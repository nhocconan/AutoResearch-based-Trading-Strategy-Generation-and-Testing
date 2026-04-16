#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Uses ATR-based trailing stop (2.5*ATR from extreme) to capture trends while limiting drawdown.
# Discrete position size 0.25. Designed to work in both bull and bear markets by requiring strong trend (ADX>25)
# and volume confirmation to avoid false reversals in ranging markets. Target: 50-150 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 1d ADX (14-period) for trend strength ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed DM and TR
    tr_sum = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_sum = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_sum = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 12h Volume Spike (volume > 1.5x 20-period average) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # === 6h ATR for trailing stop ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state, entry price, and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0  # highest price for long, lowest for short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_6h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        adx_val = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update extreme price (highest high)
            if price > extreme_price:
                extreme_price = price
            # Trailing stop: 2.5*ATR below extreme price
            if price < extreme_price - 2.5 * atr_val:
                exit_signal = True
            # Exit if Williams %R reaches overbought (-20) in strong trend
            elif williams_r[i] > -20 and adx_val > 30:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update extreme price (lowest low)
            if price < extreme_price:
                extreme_price = price
            # Trailing stop: 2.5*ATR above extreme price
            if price > extreme_price + 2.5 * atr_val:
                exit_signal = True
            # Exit if Williams %R reaches oversold (-80) in strong trend
            elif williams_r[i] < -80 and adx_val > 30:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R oversold (<-80) AND trending (ADX>25) AND volume spike
            if williams_r[i] < -80 and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_price = price
            
            # SHORT: Williams %R overbought (>-20) AND trending (ADX>25) AND volume spike
            elif williams_r[i] > -20 and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR_1dADX_12hVolumeSpike_V1"
timeframe = "6h"
leverage = 1.0