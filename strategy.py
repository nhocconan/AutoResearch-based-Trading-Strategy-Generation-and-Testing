#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Primary: 4h price breaking above/below 20-period Donchian channels
# - HTF filter: 1d volume > 1.5x 20-period MA for institutional participation
# - Regime filter: 1d Choppiness Index(14) between 38.2 and 61.8 to avoid extreme chop/trend exhaustion
# - Entry: Long when close > upper Donchian + volume filter + chop regime; Short when close < lower Donchian + volume filter + chop regime
# - Exit: Close crosses 10-period EMA (mean reversion) or chop regime shifts to extreme (>61.8 or <38.2)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# - Works in bull/bear: Donchian breakouts capture sustained moves, volume confirms validity, chop filter avoids false signals in ranging markets

name = "4h_1d_donchian_volume_chop_v1"
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
    
    # Calculate primary Donchian channels (20-period)
    def calculate_donchian(high, low, window=20):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low)
    
    # Calculate 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d Choppiness Index(14)
    def calculate_choppiness(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Sum of TR over window
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Avoid division by zero
        denominator = hh - ll
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        # Choppiness Index
        chop = 100 * np.log10(tr_sum / denominator) / np.log10(window)
        return chop
    
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Regime filter: chop between 38.2 and 61.8 to avoid extreme chop/trend exhaustion
        chop_regime_ok = (chop_1d_aligned[i] >= 38.2) & (chop_1d_aligned[i] <= 61.8)
        chop_extreme = (chop_1d_aligned[i] < 38.2) | (chop_1d_aligned[i] > 61.8)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: close > upper Donchian + volume confirmation + chop regime OK
            if (close[i] > donchian_upper[i] and volume_confirm and chop_regime_ok):
                position = 1
                signals[i] = 0.25
            # Short entry: close < lower Donchian + volume confirmation + chop regime OK
            elif (close[i] < donchian_lower[i] and volume_confirm and chop_regime_ok):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses 10-period EMA OR chop regime shifts to extreme
            if position == 1:  # Long position
                if close[i] < ema_10[i] or chop_extreme:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > ema_10[i] or chop_extreme:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals