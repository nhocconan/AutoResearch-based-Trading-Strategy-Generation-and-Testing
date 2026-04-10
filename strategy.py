#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d ADX > 25 AND volume > 1.5x avg
# - Short when price breaks below Donchian(20) low AND 1d ADX > 25 AND volume > 1.5x avg
# - Exit when price crosses Donchian(20) midline or ADX < 20 (trend weakening)
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Targets ~12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Donchian channels provide clear breakout levels with built-in stop via opposite band
# - ADX filter ensures we only trade strong trends, reducing whipsaws
# - Volume confirmation validates breakout strength
# - Works in both bull (upward breakouts) and bear (downward breakouts) markets

name = "12h_1d_donchian_adx_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midline: average of high and low bands
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high AND strong trend AND volume spike
            if (close[i] > donchian_high[i] and 
                adx_1d_aligned[i] > 25 and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.30
            # Short signal: price breaks below Donchian low AND strong trend AND volume spike
            elif (close[i] < donchian_low[i] and 
                  adx_1d_aligned[i] > 25 and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.30
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian midline (trend weakening/reversal)
            # 2. ADX drops below 20 (trend losing strength)
            if position == 1:
                if close[i] < donchian_mid[i] or adx_1d_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30  # Hold long
            elif position == -1:
                if close[i] > donchian_mid[i] or adx_1d_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30  # Hold short
    
    return signals