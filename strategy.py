#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ATR-based trend filter
# - Primary: 4h price breaking above/below Donchian(20) channel for trend continuation
# - HTF trend: 1d ADX(14) > 25 ensures strong trend (works in bull/bear regimes)
# - HTF volume: 1d volume > 1.5x 20-period MA for institutional participation
# - Entry: Long when price > Donchian upper + ADX>25 + volume spike; Short when price < Donchian lower + ADX>25 + volume spike
# - Exit: Price crosses Donchian midpoint (mean of channel) or ADX < 20 (trend weakening)
# - Position sizing: 0.30 (discrete level to balance return and drawdown)
# - Target: 80-160 total trades over 4 years (20-40/year) for 4h timeframe
# - Works in bull/bear: Donchian breakouts capture trends, ADX filter avoids chop, volume confirms participation

name = "4h_1d_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
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
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channel (20)
    def calculate_donchian(high, low, window=20):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low)
    
    # Calculate 1d ADX(14)
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                          np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+
        tr_smooth = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=window, adjust=False, min_periods=window).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 = strong trend
        strong_trend = adx_1d_aligned[i] > 25
        # Weak trend filter: ADX < 20 = trend weakening (exit)
        weak_trend = adx_1d_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper + strong trend + volume spike
            if (close[i] > donchian_upper[i] and strong_trend and volume_confirm):
                position = 1
                signals[i] = 0.30
            # Short entry: price < Donchian lower + strong trend + volume spike
            elif (close[i] < donchian_lower[i] and strong_trend and volume_confirm):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Donchian midpoint OR ADX < 20 (trend weakening)
            if position == 1:  # Long position
                if close[i] < donchian_middle[i] or weak_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1 (Short position)
                if close[i] > donchian_middle[i] or weak_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals