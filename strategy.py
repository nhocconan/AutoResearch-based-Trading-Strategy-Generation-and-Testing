#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# - Primary: 4h price breaks above Donchian(20) high for long, below Donchian(20) low for short
# - HTF volume filter: 1d volume > 1.8x 20-period MA for institutional participation
# - HTF trend filter: 1w ADX(14) > 25 indicates strong trend (use direction from 1w EMA20)
# - Entry: Long when breakout up + volume filter + 1w uptrend; Short when breakdown down + volume filter + 1w downtrend
# - Exit: Opposite Donchian(10) breakout (exit long on break below Donchian(10) low, exit short on break above Donchian(10) high)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# - Works in bull/bear: Donchian captures breakouts, volume confirms validity, ADX ensures trending environment

name = "4h_1d_1w_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) channels on 4h
    def calculate_donchian(high, low, lookback=20):
        upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
        return upper, lower
    
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, 20)
    donchian_10_upper, donchian_10_lower = calculate_donchian(high, low, 10)
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1w ADX(14) and EMA(20) for trend direction
    def calculate_adx(high, low, close, lookback=14):
        # True Range
        tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
        tr2 = np.abs(pd.Series(high).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
        tr3 = np.abs(pd.Series(low).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.where(np.isnan(tr), 0, tr)
        
        # Directional Movement
        dm_plus = np.where((pd.Series(high).diff().values > pd.Series(low).diff().values * -1) & 
                          (pd.Series(high).diff().values > 0), pd.Series(high).diff().values, 0)
        dm_minus = np.where((pd.Series(low).diff().values * -1 > pd.Series(high).diff().values) & 
                           (pd.Series(low).diff().values < 0), pd.Series(low).diff().values * -1, 0)
        
        # Smoothed values
        tr_ma = pd.Series(tr).ewm(span=lookback, min_periods=lookback, adjust=False).mean().values
        dm_plus_ma = pd.Series(dm_plus).ewm(span=lookback, min_periods=lookback, adjust=False).mean().values
        dm_minus_ma = pd.Series(dm_minus).ewm(span=lookback, min_periods=lookback, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_ma / tr_ma
        di_minus = 100 * dm_minus_ma / tr_ma
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx = np.where(np.isnan(dx), 0, dx)
        adx = pd.Series(dx).ewm(span=lookback, min_periods=lookback, adjust=False).mean().values
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_20_upper[i]) or np.isnan(donchian_20_lower[i]) or
            np.isnan(donchian_10_upper[i]) or np.isnan(donchian_10_lower[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.8 * volume_ma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend, use EMA20 for direction
        trend_strong = adx_1w_aligned[i] > 25
        trend_up = close_1w[-1] > ema_20_1w[-1] if len(close_1w) > 0 and len(ema_20_1w) > 0 else False
        trend_down = close_1w[-1] < ema_20_1w[-1] if len(close_1w) > 0 and len(ema_20_1w) > 0 else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: break above Donchian(20) upper + volume confirmation + 1w uptrend
            if (close[i] > donchian_20_upper[i] and volume_confirm and trend_strong and trend_up):
                position = 1
                signals[i] = 0.25
            # Short entry: break below Donchian(20) lower + volume confirmation + 1w downtrend
            elif (close[i] < donchian_20_lower[i] and volume_confirm and trend_strong and trend_down):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: opposite Donchian(10) breakout
            if position == 1:  # Long position
                if close[i] < donchian_10_lower[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > donchian_10_upper[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals