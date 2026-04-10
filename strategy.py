#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 1w volume confirmation and 1w ADX trend filter
# - Entry: Long when price breaks above 20-day high + 1w volume > 1.5x 10-period average + 1w ADX > 25
#          Short when price breaks below 20-day low + 1w volume > 1.5x 10-period average + 1w ADX > 25
# - Exit: Close-based reversal - exit long when price < 20-day low, exit short when price > 20-day high
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Donchian channels from daily data for structure, weekly volume for confirmation, weekly ADX for trend filter
# - Target: 20-60 trades/year (80-240 total over 4 years) to stay within HARD MAX: 150 total
# - Weekly timeframe provides more reliable trend/volume signals for daily breakout trading
# - Reduced volume threshold (1.5x vs 2.0x) and ADX threshold (25 vs 30) to increase trade frequency while maintaining edge
# - Daily timeframe balances trade frequency and fee drag for optimal test performance

name = "1d_1w_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute daily OHLC
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Pre-compute 1w volume for confirmation
    volume_1w = df_1w['volume'].values
    
    # Pre-compute 1w OHLC for Donchian and ADX calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-day Donchian channels (based on previous 20 days)
    # Upper band = highest high of previous 20 periods
    # Lower band = lowest low of previous 20 periods
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1w volume moving average (10-period)
    volume_ma_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF data to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_10_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current daily close
        close_price = close_1d[i]
        
        # Get current 1w volume for confirmation (need to align raw volume)
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_confirmation = volume_1w_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        # Trend filter: 1w ADX > 25 indicates strong trending market
        trend_filter = adx_1w_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + volume confirmation + strong trending market
            if (close_price > donchian_upper_aligned[i] and 
                volume_confirmation and 
                trend_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower + volume confirmation + strong trending market
            elif (close_price < donchian_lower_aligned[i] and 
                  volume_confirmation and 
                  trend_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < Donchian lower
            # Exit short when price > Donchian upper
            if position == 1:
                if close_price < donchian_lower_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > donchian_upper_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals