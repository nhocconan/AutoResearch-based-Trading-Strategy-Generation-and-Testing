#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and ADX regime filter.
# Long when price breaks above Camarilla R4 AND 1d volume > 2x 20-period average AND 1d ADX > 25 (trending).
# Short when price breaks below Camarilla S4 AND 1d volume > 2x 20-period average AND 1d ADX > 25.
# Exit on opposite Camarilla break (R3/S3) or when ADX < 20 (range regime).
# Uses discrete position size 0.25. Designed to capture strong breakouts in trending markets while avoiding false breakouts in ranging markets.
# Works in both bull and bear markets by requiring ADX trend filter and volume confirmation, avoiding low-probability breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (based on prior bar) ===
    # Camarilla levels calculated from prior bar's high, low, close
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = prior_high[1]  # avoid NaN at index 0
    prior_low[0] = prior_low[1]
    prior_close[0] = prior_close[1]
    
    camarilla_r4 = prior_close + ((prior_high - prior_low) * 1.1 / 2)
    camarilla_r3 = prior_close + ((prior_high - prior_low) * 1.1 / 4)
    camarilla_s3 = prior_close - ((prior_high - prior_low) * 1.1 / 4)
    camarilla_s4 = prior_close - ((prior_high - prior_low) * 1.1 / 2)
    
    # === 1d Indicators: ADX for regime filter ===
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
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: Volume Spike ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_s4[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        adx_val = adx_1d_aligned[i]
        vol_spike = volume_spike_1d_aligned[i] > 0.5  # convert back to boolean
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla R3 (take profit) or ADX < 20 (regime change to range)
            if price < camarilla_r3[i] or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla S3 (take profit) or ADX < 20 (regime change to range)
            if price > camarilla_s3[i] or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R4 AND ADX > 25 (trending) AND volume spike
            if price > camarilla_r4[i] and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S4 AND ADX > 25 (trending) AND volume spike
            elif price < camarilla_s4[i] and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_VolumeSpike_ADXFilter_V1"
timeframe = "6h"
leverage = 1.0