#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h volume confirmation and 1d ADX regime filter
# - Primary: 1h price breaking above/below 20-period Donchian channels captures breakouts
# - Volume filter: 4h volume > 1.8x 20-period volume MA confirms institutional participation
# - Regime filter: 1d ADX(14) > 20 ensures trending market, avoids whipsaws in ranging markets
# - Exit: Price reverses back to opposite Donchian channel level (midpoint for re-entry prevention)
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Session filter: 08-20 UTC to reduce noise trades during low liquidity periods
# - Works in bull/bear: Donchian adapts to volatility, volume confirms breakout validity, ADX filters weak trends
# - Target: 80-120 total trades over 4 years = 20-30/year for 1h timeframe

name = "1h_4h_1d_donchian_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1h Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_ma_20
    donchian_low = low_ma_20
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 4h volume spike filter: volume > 1.8x 20-period volume MA
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    
    # Calculate 1d ADX(14) for regime filter
    high_diff_1d = high_1d - np.roll(high_1d, 1)
    low_diff_1d = np.roll(low_1d, 1) - low_1d
    close_diff_1d = np.roll(close_1d, 1) - close_1d
    high_diff_1d[0] = 0
    low_diff_1d[0] = 0
    close_diff_1d[0] = 0
    
    plus_dm_1d = np.where((high_diff_1d > low_diff_1d) & (high_diff_1d > 0), high_diff_1d, 0)
    minus_dm_1d = np.where((low_diff_1d > high_diff_1d) & (low_diff_1d > 0), low_diff_1d, 0)
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = high_1d[0] - low_1d[0]
    tr2_1d[0] = np.abs(high_1d[0] - close_1d[0])
    tr3_1d[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_14_1d = pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).mean().values
    minus_dm_14_1d = pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).mean().values
    
    plus_di_1d = np.where(atr_14_1d > 0, 100 * plus_dm_14_1d / atr_14_1d, 0)
    minus_di_1d = np.where(atr_14_1d > 0, 100 * minus_dm_14_1d / atr_14_1d, 0)
    
    dx_1d = np.where((plus_di_1d + minus_di_1d) > 0, 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma_20_4h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.8x 20-period volume MA
        volume_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)
        vol_spike = volume_4h_current[i] > 1.8 * volume_ma_20_4h_aligned[i]
        
        # Regime filter: ADX > 20 to ensure trending conditions
        strong_trend = adx_1d_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian High + vol spike + strong trend + session
            if (close[i] > donchian_high[i] and 
                vol_spike and strong_trend):
                position = 1
                signals[i] = 0.20
            # Short entry: price breaks below Donchian Low + vol spike + strong trend + session
            elif (close[i] < donchian_low[i] and 
                  vol_spike and strong_trend):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price reverses back to Donchian midpoint (prevents immediate re-entry)
            if position == 1:  # Long position
                if close[i] < donchian_mid[i]:  # Exit when price crosses below midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if close[i] > donchian_mid[i]:  # Exit when price crosses above midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals