#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and 12h volume confirmation
# - Primary: 4h price breaks above Donchian(20) high (long) or below Donchian(20) low (short)
# - Volume filter: 12h volume > 1.5x 20-period volume MA to confirm institutional participation
# - Trend filter: 1d ADX(14) > 25 to ensure we're in a trending market (avoid chop/ranges)
# - Entry: Long when breakout above upper band + volume spike + ADX > 25
#          Short when breakout below lower band + volume spike + ADX > 25
# - Exit: Close crosses back inside Donchian channel (mean reversion exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# - Works in bull/bear: Donchian breakouts capture trends, volume spike avoids fakeouts, ADX filter avoids false signals in ranging markets

name = "4h_12h_1d_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume MA(20) for volume spike filter
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 1d ADX(14)
    # ADX = 100 * smoothed DX, where DX = |+DI - -DI| / (+DI + -DI)
    # +DI = 100 * smoothed +DM / ATR, -DI = 100 * smoothed -DM / ATR
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = high_low_1d[0]
    low_close_1d[0] = high_low_1d[0]
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    
    # +DM and -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    atr_14_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14_1d = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14_1d = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di_14_1d = 100 * plus_dm_14_1d / atr_14_1d
    minus_di_14_1d = 100 * minus_dm_14_1d / atr_14_1d
    
    # DX and ADX
    dx_14_1d = 100 * np.abs(plus_di_14_1d - minus_di_14_1d) / (plus_di_14_1d + minus_di_14_1d)
    dx_14_1d = np.where((plus_di_14_1d + minus_di_14_1d) == 0, 0, dx_14_1d)
    adx_14_1d = pd.Series(dx_14_1d).ewm(alpha=1/14, adjust=False).mean().values
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(30, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike filter: current 12h volume > 1.5x 20-period volume MA
        volume_spike = volume_12h[i // 12] > 1.5 * vol_ma_20_12h_aligned[i] if i // 12 < len(volume_12h) else False
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_14_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: breakout above upper Donchian band + volume spike + ADX > 25
            if (close[i] > highest_20[i] and volume_spike and trend_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: breakout below lower Donchian band + volume spike + ADX > 25
            elif (close[i] < lowest_20[i] and volume_spike and trend_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside Donchian channel (mean reversion exit)
            if position == 1:  # Long position
                if close[i] < lowest_20[i]:  # Exit when price breaks below lower band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > highest_20[i]:  # Exit when price breaks above upper band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals