#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d volume spike and 1w ADX trend filter
# - Long when Williams %R(14) crosses above -80 (oversold bounce) + volume > 2.0x 20-period 1d volume SMA + ADX(14) > 25 (trending market)
# - Short when Williams %R(14) crosses below -20 (overbought rejection) + volume > 2.0x 20-period 1d volume SMA + ADX(14) > 25 (trending market)
# - Exit: Williams %R crosses -50 (mean reversion to midpoint)
# - Position sizing: 0.25 discrete level
# - Targets ~20-30 trades/year on 4h timeframe. Williams %R captures momentum reversals,
#   volume confirmation validates conviction, ADX filter ensures we trade with trend strength.
#   Works in bull/bear: oversold bounces in bear markets, overbought rejections in bull markets,
#   ADX filter avoids choppy markets where Williams %R gives false signals.

name = "4h_1d_1w_williamsr_volume_adx_v1"
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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R(14) from 4h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    plus_di = np.where(tr_14 == 0, 0, plus_di)
    minus_di = np.where(tr_14 == 0, 0, minus_di)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_aligned[i] > 25
        
        # Williams %R entry conditions
        # Long: Williams %R crosses above -80 (oversold bounce) + volume confirmation + trending market
        # Short: Williams %R crosses below -20 (overbought rejection) + volume confirmation + trending market
        long_entry = (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                     vol_confirm and 
                     trend_filter)
        short_entry = (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                      vol_confirm and 
                      trend_filter)
        
        # Exit conditions: Williams %R crosses -50 (mean reversion)
        exit_long = williams_r[i] < -50 and williams_r[i-1] >= -50
        exit_short = williams_r[i] > -50 and williams_r[i-1] <= -50
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals