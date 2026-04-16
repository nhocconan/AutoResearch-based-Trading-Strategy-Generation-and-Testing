#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R reversal with 1w ADX trend filter and volume confirmation.
# Williams %R(14) < -80 indicates oversold (long setup), > -20 indicates overbought (short setup).
# Requires 1w ADX > 25 to ensure trending environment (avoid whipsaws in ranging markets).
# Volume must be > 1.8x 20-period 1d average to confirm conviction.
# Long when Williams %R crosses above -80 from below AND volume spike AND 1w ADX > 25.
# Short when Williams %R crosses below -20 from above AND volume spike AND 1w ADX > 25.
# Exit when Williams %R crosses above -20 (for longs) or below -80 (for shorts) or ATR-based stoploss (2.5*ATR).
# Uses discrete position size 0.25. Designed to capture mean reversals within strong trends.
# Works in both bull and bear markets by requiring 1w trend filter, avoiding false reversals in chop.
# Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag while capturing high-probability reversals.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Williams %R (14-period) ===
    highest_high_1d = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_1d - close) / (highest_high_1d - lowest_low_1d)
    
    # === 1d Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_1d)
    
    # === 1w Indicators: ADX > 25 (strong trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1w).diff()
    dm_minus = pd.Series(low_1w).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    strong_trend = adx_aligned > 25
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for Williams %R/ADX)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 1d ATR for stoploss
    tr1_1d = pd.Series(high).diff()
    tr2_1d = pd.Series(low).diff().abs()
    tr3_1d = pd.Series(close).shift(1).diff().abs()
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or np.isnan(strong_trend[i]) or
            np.isnan(atr_1d_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        atr_val = atr_1d_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -20 (overbought)
            if williams_r[i] >= -20:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -80 (oversold)
            if williams_r[i] <= -80:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Williams %R crossover signals
            wr_cross_above_80 = williams_r[i] > -80 and williams_r[i-1] <= -80
            wr_cross_below_20 = williams_r[i] < -20 and williams_r[i-1] >= -20
            
            # LONG: Williams %R crosses above -80 from below (ending oversold) AND volume spike AND strong 1w trend
            if wr_cross_above_80 and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R crosses below -20 from above (ending overbought) AND volume spike AND strong 1w trend
            elif wr_cross_below_20 and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_WilliamsR_1dVolumeSpike_1wADX_V1"
timeframe = "1d"
leverage = 1.0