#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_DonchianBreakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for higher timeframe trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12-period Donchian channels on 12h data
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    
    # Shift to avoid look-ahead: use previous bar's extreme
    donchian_upper = np.roll(highest_high, 1)
    donchian_lower = np.roll(lowest_low, 1)
    donchian_upper[0] = np.nan
    donchian_lower[0] = np.nan
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, volatility low, uptrend, volume confirmation
            long_cond = (close[i] > donchian_upper[i] and
                        atr_14_aligned[i] < np.nanmedian(atr_14_aligned[max(0, i-50):i]) * 1.5 and
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_confirm[i])
            
            # Short: price breaks below Donchian lower, volatility low, downtrend, volume confirmation
            short_cond = (close[i] < donchian_lower[i] and
                         atr_14_aligned[i] < np.nanmedian(atr_14_aligned[max(0, i-50):i]) * 1.5 and
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_confirm[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower or trend reverses
            if close[i] < donchian_lower[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper or trend reverses
            if close[i] > donchian_upper[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals