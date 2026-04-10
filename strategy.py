#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 12h ADX trend filter
# - Entry: Long when price breaks above 4h Donchian upper channel + 1d volume > 1.8x 20-period average + 12h ADX > 25
#          Short when price breaks below 4h Donchian lower channel + 1d volume > 1.8x 20-period average + 12h ADX > 25
# - Exit: Close-based reversal - exit long when price < 4h Donchian middle (10-period), exit short when price > 4h Donchian middle
# - Position sizing: 0.30 (discrete level to balance return and fee drag)
# - Uses 4h price structure for precise timing, daily volume for institutional confirmation, 12h ADX for medium-term trend
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within HARD MAX: 400 total
# - Volume threshold increased to 1.8x to reduce false breakouts, ADX threshold raised to 25 for stronger trend filter
# - Middle band exit provides smoother exits than pure breakout reversal, reducing whipsaw

name = "4h_1d_12h_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 12h OHLC for ADX calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper channel = highest high over last 20 periods
    # Lower channel = lowest low over last 20 periods
    # Middle channel = (upper + lower) / 2
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    
    donchian_upper = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series_4h.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ADX (14-period)
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    minus_di_12h = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_4h, 'low': low_4h, 'close': close_4h}), donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_4h, 'low': low_4h, 'close': close_4h}), donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_4h, 'low': low_4h, 'close': close_4h}), donchian_middle)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Volume confirmation: current 1d volume > 1.8x 20-period average
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 1.8 * volume_ma_aligned[i]
        
        # Trend filter: 12h ADX > 25 indicates strong trending market
        trend_filter = adx_12h_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + volume confirmation + strong trending market
            if (close_price > donchian_upper_aligned[i] and 
                volume_confirmation and 
                trend_filter):
                position = 1
                signals[i] = 0.30
            # Short entry: price breaks below Donchian lower + volume confirmation + strong trending market
            elif (close_price < donchian_lower_aligned[i] and 
                  volume_confirmation and 
                  trend_filter):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < Donchian middle
            # Exit short when price > Donchian middle
            if position == 1:
                if close_price < donchian_middle_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if close_price > donchian_middle_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals