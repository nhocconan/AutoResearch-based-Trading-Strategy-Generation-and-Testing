#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 1.5x 20-period average AND chop > 61.8 (ranging).
# Short when price breaks below Camarilla S3 AND 1d volume > 1.5x 20-period average AND chop > 61.8 (ranging).
# Uses discrete position size 0.25. Camarilla levels provide precise intraday support/resistance,
# volume spike confirms institutional participation, chop filter avoids trending markets where pivots fail.
# Designed to work in ranging markets which dominate BTC/ETH during 2022-2024 and 2025+.
# Target: 100-180 trades over 4 years (25-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Typical Price for Pivot Calculation ===
    typical_price = (high + low + close) / 3.0
    
    # Get 1d data once before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Previous day's Camarilla levels ===
    # Camarilla pivot levels based on previous day's OHLC
    # R4 = close + ((high - low) * 1.5000)
    # R3 = close + ((high - low) * 1.2500)
    # R2 = close + ((high - low) * 1.1666)
    # R1 = close + ((high - low) * 1.0833)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.0833)
    # S2 = close - ((high - low) * 1.1666)
    # S3 = close - ((high - low) * 1.2500)
    # S4 = close - ((high - low) * 1.5000)
    
    # Calculate using previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have rolled values from last day - set to NaN to avoid false signals
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    
    camarilla_r3 = camarilla_pp + (camarilla_range * 1.2500)
    camarilla_s3 = camarilla_pp - (camarilla_range * 1.2500)
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # === 1d Indicators: Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log(n))) / log10(n)
    # Where n = 14, CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum()
    n = 14
    log_n = np.log10(n)
    chop = 100 * (np.log10(sum_atr_14 / (n * log_n)) / log_n)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ATR, 20 for volume MA, 14 for CHOP)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike_1d_aligned[i] > 0.5  # Convert back to boolean
        chop_val = chop_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to pivot point or volume spike ends
            if price <= camarilla_pp_aligned[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to pivot point or volume spike ends
            if price >= camarilla_pp_aligned[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R3 AND volume spike AND chop > 61.8 (ranging)
            if price > r3_level and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S3 AND volume spike AND chop > 61.8 (ranging)
            elif price < s3_level and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_CamarillaR3S3_1dVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0