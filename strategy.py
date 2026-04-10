#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and 1d ADX regime filter
# - Primary: 4h price touching Camarilla H3/L3 levels captures institutional breakout/retest patterns
# - Volume filter: 12h volume > 1.8x 20-period volume MA confirms strong participation
# - Regime filter: 1d ADX(14) > 25 ensures sufficient trend strength (avoids whipsaws in ranging markets)
# - Exit: Price crosses opposite Camarilla H4/L4 level for defined risk/reward
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# - Works in bull/bear: Camarilla levels adapt to volatility, volume confirms strength, ADX avoids weak trends

name = "4h_12h_1d_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), etc.
    # We need previous day's data for today's levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Calculate Camarilla levels for 1d timeframe
    rangew = prev_high_1d - prev_low_1d
    camarilla_h4 = prev_close_1d + 1.5 * rangew
    camarilla_h3 = prev_close_1d + 1.25 * rangew
    camarilla_l3 = prev_close_1d - 1.25 * rangew
    camarilla_l4 = prev_close_1d - 1.5 * rangew
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 12h volume MA(20) for volume filter
    volume_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20)
    
    # Calculate 1d ADX(14) for regime filter
    high_diff = high_1d - np.roll(high_1d, 1)
    low_diff = np.roll(low_1d, 1) - low_1d
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = np.where(atr_14 > 0, 100 * plus_dm_14 / atr_14, 0)
    minus_di = np.where(atr_14 > 0, 100 * minus_dm_14 / atr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_ma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1.8x 20-period volume MA (strong confirmation)
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_confirmed = volume_12h_aligned[i] > 1.8 * volume_ma_20_aligned[i]
        
        # Regime filter: ADX > 25 to ensure trending conditions
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price crosses above Camarilla H3 + volume confirmation + trending
            if (close[i] > camarilla_h3_aligned[i] and close[i-1] <= camarilla_h3_aligned[i-1] and 
                volume_confirmed and trending):
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below Camarilla L3 + volume confirmation + trending
            elif (close[i] < camarilla_l3_aligned[i] and close[i-1] >= camarilla_l3_aligned[i-1] and 
                  volume_confirmed and trending):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price crosses opposite Camarilla H4/L4 level
            if position == 1:  # Long position
                if close[i] < camarilla_l4_aligned[i]:  # Exit when price crosses below Camarilla L4
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > camarilla_h4_aligned[i]:  # Exit when price crosses above Camarilla H4
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals