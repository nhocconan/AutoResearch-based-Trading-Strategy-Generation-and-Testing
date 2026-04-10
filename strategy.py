#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX regime filter
# - Entry: Long when price breaks above Donchian upper band (20-period high) + 1d volume > 1.5x 20-period average + ADX(14) < 25 (range/low trend)
#          Short when price breaks below Donchian lower band (20-period low) + 1d volume > 1.5x 20-period average + ADX(14) < 25
# - Exit: ATR(14) trailing stop (2.5x) on 4h timeframe
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses 1d for volume confirmation to avoid lower timeframe noise
# - Donchian provides clear structure, volume confirms conviction, ADX filter avoids choppy whipsaws
# - Target: 25-50 trades/year (100-200 total over 4 years) to stay within HARD MAX: 400 total

name = "4h_1d_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d OHLC and volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels for 4h timeframe (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume and its 20-period moving average for confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Pre-compute 1d ADX(14) for regime filter
    # ADX calculation: +DM, -DM, TR, then DX, then smoothed ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    plus_dm = high_1d - np.roll(high_1d, 1)
    minus_dm = np.roll(low_1d, 1) - low_1d
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_smooth = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d_smooth
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d_smooth
    
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 4h ATR(14) for trailing stop
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr1_4h[0] = np.nan
    tr2_4h[0] = np.nan
    tr3_4h[0] = np.nan
    tr_4h = np.maximum.reduce([tr1_4h, tr2_4h, tr3_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(atr_4h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 4h close for breakout detection
        close_price = close_4h[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmation = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        # ADX filter: only trade when ADX < 25 (range/low trend environment)
        adx_filter = adx_aligned[i] < 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price closes above Donchian high with volume confirmation and ADX filter
            if close_price > donchian_high[i] and volume_confirmation and adx_filter:
                position = 1
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short breakout: price closes below Donchian low with volume confirmation and ADX filter
            elif close_price < donchian_low[i] and volume_confirmation and adx_filter:
                position = -1
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_4h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_4h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals