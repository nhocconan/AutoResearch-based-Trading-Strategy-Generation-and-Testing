#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day volatility filter and volume confirmation
# Uses Donchian(20) from daily data for entry signals
# Daily ATR-based volatility filter to avoid choppy markets (ATR > 20-period median)
# Volume confirmation > 1.3x 20-period EMA to reduce false breakouts
# Designed for 20-40 trades/year with clear breakout logic
# Works in bull markets via upward breaks and in bear markets via downward breaks
# Position size: 0.25 to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once for Donchian channels and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels from daily high/low (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper/lower bands (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR for volatility filter (14-period)
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    close_1d_shift[0] = close_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_median = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned daily Donchian levels
        donchian_high_i = align_htf_to_ltf(prices, df_1d, donchian_high)[i]
        donchian_low_i = align_htf_to_ltf(prices, df_1d, donchian_low)[i]
        
        # Get aligned daily volatility filter
        atr_med_i = align_htf_to_ltf(prices, df_1d, atr_median)[i]
        
        if np.isnan(donchian_high_i) or np.isnan(donchian_low_i) or np.isnan(atr_med_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volatility filter: only trade when ATR > median (avoid choppy markets)
        vol_filter = atr_1d[i] > atr_med_i
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Long: Price breaks above Donchian high + volatility filter + volume
        if position == 0 and close[i] > donchian_high_i and vol_filter and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Price breaks below Donchian low + volatility filter + volume
        elif position == 0 and close[i] < donchian_low_i and vol_filter and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Price returns to opposite Donchian level or volatility drops
        elif position != 0:
            if position == 1 and (close[i] < donchian_low_i or not vol_filter):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > donchian_high_i or not vol_filter):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_DailyVolatility_Volume"
timeframe = "4h"
leverage = 1.0