#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and 1w ADX trend filter
# - Primary: 4h price breaking above/below Camarilla H3/L3 levels from 1d HTF
# - Volume filter: 1d volume > 2.0x 20-period volume MA to ensure institutional participation
# - Trend filter: 1w ADX > 25 to ensure trending market (avoid chop/range)
# - Exit: Price reverses back to Camarilla H4/L4 levels (stronger reversal signal)
# - Position sizing: 0.25 (discrete level to minimize fee churn while maintaining edge)
# - Target: 80-160 total trades over 4 years = 20-40/year for 4h timeframe
# - Works in bull/bear: Camarilla adapts to volatility, volume confirms breakout strength, ADX filter avoids false signals in ranging markets

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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    #            L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First value
    prev_high_1d = np.roll(high_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d = np.roll(low_1d, 1)
    prev_low_1d[0] = low_1d[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_h3 = prev_close_1d + 1.1 * camarilla_range / 4
    camarilla_l3 = prev_close_1d - 1.1 * camarilla_range / 4
    camarilla_h4 = prev_close_1d + 1.1 * camarilla_range / 2
    camarilla_l4 = prev_close_1d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d volume confirmation: volume > 2.0x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1w ADX for trend filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    high_low_1w = high_1w - low_1w
    high_close_1w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_1w = np.abs(low_1w - np.roll(close_1w, 1))
    
    high_low_1w[0] = high_1w[0] - low_1w[0]
    high_close_1w[0] = np.abs(high_1w[0] - close_1w[0])
    low_close_1w[0] = np.abs(low_1w[0] - close_1w[0])
    
    tr_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    
    # +DM and -DM
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_smoothed = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    plus_dm_smoothed = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2.0x 20-period volume MA
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = volume_1d_current[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 to ensure trending market
        trending_market = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + vol confirmation + trending market
            if (close[i] > camarilla_h3_aligned[i] and 
                vol_confirm and trending_market):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + vol confirmation + trending market
            elif (close[i] < camarilla_l3_aligned[i] and 
                  vol_confirm and trending_market):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price returns to Camarilla H4/L4 levels (stronger reversal signal)
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