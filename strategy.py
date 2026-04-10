#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d ADX trend filter and volume spike
# - Long when Williams %R(14) < -80 (oversold) + ADX(14) > 25 (trending) + volume > 1.5x 20-period 1d volume SMA
# - Short when Williams %R(14) > -20 (overbought) + ADX(14) > 25 (trending) + volume > 1.5x 20-period 1d volume SMA
# - Exit: Williams %R returns to -50 (mean reversion to equilibrium)
# - Position sizing: 0.25 discrete level
# - Williams %R identifies extreme momentum exhaustion
# - ADX ensures we only trade in trending markets to avoid false reversals in ranging conditions
# - Volume confirmation adds conviction to the mean reversion signal
# - Works in bull/bear: mean reversion occurs in all regimes, ADX filter prevents chop whipsaws

name = "4h_1d_williamsr_adx_volume_v1"
timeframe = "4h"
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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R on 1d timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    hh_ll = highest_high_14 - lowest_low_14
    williams_r_1d = np.where(hh_ll != 0, (highest_high_14 - close_1d) / hh_ll * -100, -50)
    
    # Align Williams %R to 4h timeframe (completed 1d bar only)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate ADX on 1d timeframe
    # ADX calculation requires +DI and -DI
    # +DI = 100 * smoothed +DM / ATR
    # -DI = 100 * smoothed -DM / ATR
    # ADX = 100 * smoothed |+DI - -DI| / (+DI + -DI)
    
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
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_1d_aligned[i] > 25
        
        # Williams %R mean reversion entry conditions
        # Long: oversold (%R < -80) + trending + volume confirmation
        # Short: overbought (%R > -20) + trending + volume confirmation
        long_entry = (williams_r_1d_aligned[i] < -80 and 
                     trend_filter and 
                     vol_confirm)
        short_entry = (williams_r_1d_aligned[i] > -20 and 
                      trend_filter and 
                      vol_confirm)
        
        # Exit conditions: Williams %R returns to -50 (mean reversion equilibrium)
        exit_long = williams_r_1d_aligned[i] > -50
        exit_short = williams_r_1d_aligned[i] < -50
        
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