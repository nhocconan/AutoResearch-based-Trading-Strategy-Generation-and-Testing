#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and 1d ADX trend filter.
# Long when price breaks above 12h Camarilla R1 AND volume > 1.5x 20-period 1d average AND 1d ADX > 20.
# Short when price breaks below 12h Camarilla S1 AND volume > 1.5x 20-period 1d average AND 1d ADX > 20.
# Exit when price crosses the 12h Camarilla midpoint (R3/S3) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture intraday breakouts with volume and trend confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.
# Works in both bull and bear markets by requiring volume confirmation and trend filter (ADX>20).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on prior 12h bar) ===
    # Calculate Camarilla levels from prior 12h bar's OHLC
    # We'll compute for each bar using the prior completed 12h bar
    # To avoid look-ahead, we shift the prior bar's OHLC by 1
    prior_open = np.roll(open_, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    # Set first value to NaN (no prior bar)
    prior_open[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Camarilla calculations
    range_ = prior_high - prior_low
    camarilla_mid = (prior_high + prior_low) / 2
    camarilla_r1 = camarilla_mid + range_ * 1.1 / 12
    camarilla_s1 = camarilla_mid - range_ * 1.1 / 12
    camarilla_r3 = camarilla_mid + range_ * 1.1 / 4
    camarilla_s3 = camarilla_mid - range_ * 1.1 / 4
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1d Indicators: ADX > 20 (trending market filter) ===
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
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    strong_trend = adx_aligned > 20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 12h ATR for stoploss (using 12h data aligned)
    # We'll approximate using 1d ATR since 12h data not directly available
    # In practice, we could use 1d ATR as proxy for volatility
    atr_12h_raw = atr_1d  # Using 1d ATR as proxy
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(camarilla_mid[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(volume_spike[i]) or np.isnan(strong_trend[i]) or np.isnan(atr_12h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Camarilla midpoint (R3/S3 midpoint)
            camarilla_mid_val = (camarilla_r3[i] + camarilla_s3[i]) / 2
            if price < camarilla_mid_val:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Camarilla midpoint (R3/S3 midpoint)
            camarilla_mid_val = (camarilla_r3[i] + camarilla_s3[i]) / 2
            if price > camarilla_mid_val:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND strong trending market
            if price > camarilla_r1[i] and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND strong trending market
            elif price < camarilla_s1[i] and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            # Hold current position
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1_S1_1dVolumeSpike_1dADX_V1"
timeframe = "12h"
leverage = 1.0