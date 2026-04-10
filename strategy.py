#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and 1w ADX regime filter
# - Primary: 4h price breaking above/below Camarilla H3/L3 levels from 1d pivot
# - Volume filter: 1d volume > 2.0x 20-period volume MA to ensure strong participation
# - Regime filter: 1w ADX(14) > 20 to ensure trending market (avoids choppy conditions)
# - Exit: Price reverses back to Camarilla H4/L4 levels
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# - Works in bull/bear: Camarilla adapts to volatility, volume confirms institutional interest, ADX filters weak/regime

name = "4h_1d_1w_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume spike filter: volume > 2.0x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1w ADX(14) for regime filter
    high_diff_1w = high_1w - np.roll(high_1w, 1)
    low_diff_1w = np.roll(low_1w, 1) - low_1w
    close_diff_1w = np.roll(close_1w, 1) - close_1w
    high_diff_1w[0] = 0
    low_diff_1w[0] = 0
    close_diff_1w[0] = 0
    
    plus_dm_1w = np.where((high_diff_1w > low_diff_1w) & (high_diff_1w > 0), high_diff_1w, 0)
    minus_dm_1w = np.where((low_diff_1w > high_diff_1w) & (low_diff_1w > 0), low_diff_1w, 0)
    
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1_1w[0] = high_1w[0] - low_1w[0]
    tr2_1w[0] = np.abs(high_1w[0] - close_1w[0])
    tr3_1w[0] = np.abs(low_1w[0] - close_1w[0])
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    
    atr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    plus_dm_14_1w = pd.Series(plus_dm_1w).rolling(window=14, min_periods=14).mean().values
    minus_dm_14_1w = pd.Series(minus_dm_1w).rolling(window=14, min_periods=14).mean().values
    
    plus_di_1w = np.where(atr_14_1w > 0, 100 * plus_dm_14_1w / atr_14_1w, 0)
    minus_di_1w = np.where(atr_14_1w > 0, 100 * minus_dm_14_1w / atr_14_1w, 0)
    
    dx_1w = np.where((plus_di_1w + minus_di_1w) > 0, 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w), 0)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2.0x 20-period volume MA
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_spike = volume_1d_current[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Regime filter: ADX > 20 to ensure trending conditions
        strong_trend = adx_1w_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + vol spike + strong trend
            if (close[i] > camarilla_h3_aligned[i] and 
                vol_spike and strong_trend):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + vol spike + strong trend
            elif (close[i] < camarilla_l3_aligned[i] and 
                  vol_spike and strong_trend):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price reverses back to Camarilla H4/L4 levels
            if position == 1:  # Long position
                if close[i] < camarilla_h4_aligned[i]:  # Exit when price crosses below H4
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > camarilla_l4_aligned[i]:  # Exit when price crosses above L4
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals