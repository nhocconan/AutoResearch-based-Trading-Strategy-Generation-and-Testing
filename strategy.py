#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX trend filter
# - Primary: 1d price breaking above/below 20-day Donchian channels
# - Volume filter: 1w volume > 2.0x 20-period volume MA to ensure institutional participation
# - Trend filter: 1w ADX > 25 to ensure trending market (avoid chop/range)
# - Exit: Price reverses back to midpoint of Donchian channel (mean reversion within trend)
# - Position sizing: 0.25 (discrete level to minimize fee churn while maintaining edge)
# - Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# - Works in bull/bear: Donchian adapts to volatility, volume confirms breakout strength, ADX filter avoids false signals in ranging markets

name = "1d_1w_donchian_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 20-day Donchian channels (based on past 20 days)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1w volume confirmation: volume > 2.0x 20-period volume MA
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Calculate 1w ADX for trend filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    high_low_1w = high_1w - low_1w
    high_close_1w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_1w = np.abs(low_1w - np.roll(close_1w, 1))
    
    # Handle first element
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
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma_20_1w_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1w volume > 2.0x 20-period volume MA
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)
        vol_confirm = volume_1w_current[i] > 2.0 * volume_ma_20_1w_aligned[i]
        
        # Trend filter: ADX > 25 to ensure trending market
        trending_market = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + vol confirmation + trending market
            if (close[i] > donchian_high[i] and 
                vol_confirm and trending_market):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + vol confirmation + trending market
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and trending_market):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price returns to Donchian midpoint (mean reversion within trend)
            if position == 1:  # Long position
                if close[i] < donchian_mid[i]:  # Exit when price crosses below midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > donchian_mid[i]:  # Exit when price crosses above midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals