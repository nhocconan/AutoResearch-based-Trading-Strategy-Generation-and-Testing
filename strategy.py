#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d regime filter (ADX) and volume confirmation
# - Donchian breakout: price breaks above/below 20-period high/low on 4h
# - Regime filter: ADX(14) > 25 on 1d for trending markets (avoid whipsaws in ranging)
# - Volume confirmation: current 4h volume > 1.3x 20-period average
# - Entry logic:
#   * Long: price > Donchian High(20) AND ADX > 25 AND volume spike
#   * Short: price < Donchian Low(20) AND ADX > 25 AND volume spike
# - Exit: ATR(14) trailing stop (2.5x) on 4h timeframe
# - Weekly trend filter: avoid counter-trend trades (long only when price > weekly EMA50)
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within HARD MAX: 400 total
# - Works in bull markets via breakouts, in bear markets via filtered shorts and regime avoidance

name = "4h_1d_donchian_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian High/Low (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    adx = np.where((di_plus + di_minus) == 0, 0, adx)  # avoid division by zero
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)  # Note: using 1d data for alignment but values are 4h
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 4h ATR for trailing stop
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr1_4h[0] = np.nan
    tr2_4h[0] = np.nan
    tr3_4h[0] = np.nan
    tr_4h = np.maximum.reduce([tr1_4h, tr2_4h, tr3_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 4h volume and its 20-period moving average
    volume_4h = prices['volume'].values
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(volume_ma_20_4h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 4h volume for filter
        volume_4h_current = volume_4h[i]
        
        # Get current price
        close_price = close_4h[i]
        
        # Donchian breakout conditions
        breakout_up = close_price > donchian_high_aligned[i]
        breakout_down = close_price < donchian_low_aligned[i]
        
        # ADX regime filter: only trade in trending markets
        trending_market = adx_aligned[i] > 25
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_spike = volume_4h_current > 1.3 * volume_ma_20_4h[i]
        
        # Weekly trend filter: avoid counter-trend trades
        weekly_uptrend = close_4h[i] > ema_50_aligned[i]
        weekly_downtrend = close_4h[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if trending_market and volume_spike:
                # Long: breakout up AND weekly uptrend (or neutral)
                if breakout_up and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: breakout down AND weekly downtrend (or neutral)
                elif breakout_down and weekly_downtrend:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
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