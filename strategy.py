#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume spike
# - Long when price breaks above 20-period 1d Donchian high + ADX(14) > 25 + volume > 1.5x 20-period 1d volume SMA
# - Short when price breaks below 20-period 1d Donchian low + ADX(14) > 25 + volume > 1.5x 20-period 1d volume SMA
# - Exit: price returns to 10-period 1d Donchian midpoint (mean reversion)
# - Position sizing: 0.30 discrete level
# - Donchian breakouts capture strong momentum moves
# - ADX ensures we only trade in trending markets to avoid false breakouts in ranging conditions
# - Volume confirmation adds conviction to the breakout signal
# - Works in bull/bear: breakouts occur in all regimes, ADX filter prevents chop whipsaws

name = "12h_1d_donchian_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Donchian channels on 1d timeframe (20-period)
    highest_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid_10 = (pd.Series(df_1d['high']).rolling(window=10, min_periods=10).max().values + 
                       pd.Series(df_1d['low']).rolling(window=10, min_periods=10).min().values) / 2
    
    # Align Donchian channels to 12h timeframe (completed 1d bar only)
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    donchian_mid_10_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_10)
    
    # Calculate ADX on 1d timeframe
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d.iloc[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]  # first bar
    
    # Directional Movement
    up_move = df_1d['high'].diff()
    down_move = df_1d['low'].diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di_1d = np.where(atr_1d != 0, plus_dm_smooth / atr_1d * 100, 0)
    minus_di_1d = np.where(atr_1d != 0, minus_dm_smooth / atr_1d * 100, 0)
    
    # ADX
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                     np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d) * 100, 0)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or 
            np.isnan(donchian_mid_10_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 12h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_1d_aligned[i] > 25
        
        # Donchian breakout entry conditions
        # Long: price breaks above 20-period Donchian high + trending + volume confirmation
        # Short: price breaks below 20-period Donchian low + trending + volume confirmation
        long_entry = (close[i] > highest_high_20_aligned[i] and 
                     trend_filter and 
                     vol_confirm)
        short_entry = (close[i] < lowest_low_20_aligned[i] and 
                      trend_filter and 
                      vol_confirm)
        
        # Exit conditions: price returns to 10-period Donchian midpoint (mean reversion)
        exit_long = close[i] < donchian_mid_10_aligned[i]
        exit_short = close[i] > donchian_mid_10_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.30
            elif short_entry:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
    
    return signals