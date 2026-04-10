#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX > 25) and volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND volume > 1.5x 20-period MA AND ADX(1d) > 25 (trending market)
# - Short when Bear Power > 0 AND volume confirmation AND ADX(1d) > 25
# - ATR(14) trailing stop (2.0x) on 6h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within HARD MAX: 300 total
# - Elder Ray works well in 6h timeframe with regime filter, effective in both bull and bear markets by measuring bull/bear strength relative to EMA

name = "6h_1d_elderray_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC for EMA and ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA(13)
    bear_power_1d = ema_13_1d - low_1d   # Bear Power = EMA(13) - Low
    
    # Align Elder Ray to 6h timeframe (completed 1d bar only)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Pre-compute 1d ADX(14) for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = np.nan
    down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_period = 14
    atr_1d = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 1d volume and its 20-day moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute 6h ATR for trailing stop
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = np.nan
    tr2_6h[0] = np.nan
    tr3_6h[0] = np.nan
    tr_6h = np.maximum.reduce([tr1_6h, tr2_6h, tr3_6h])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_6h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 1d volume for filter (use raw volume, aligned)
        volume_1d_current = volume_1d
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        # Regime filter: ADX > 25 indicates trending market
        trending_market = adx_aligned[i] > 25.0
        
        close_price = close_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND volume confirmation AND trending market
            if bull_power_aligned[i] > 0 and volume_confirm and trending_market:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND volume confirmation AND trending market
            elif bear_power_aligned[i] > 0 and volume_confirm and trending_market:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.0*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.0 * atr_6h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.0*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.0 * atr_6h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals