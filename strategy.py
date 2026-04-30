#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation
# Camarilla R3/S3 levels provide strong intraday support/resistance for breakouts
# 1d ADX > 25 ensures we only trade in trending markets, reducing false signals
# Volume spike (2.0x 50-period average) confirms institutional participation
# Works in bull markets via breakouts above R3 and bear markets via breakdowns below S3
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_1dADX25_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load daily data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d ADX for trend filter
    plus_dm = pd.Series(high_1d).diff()
    minus_dm = pd.Series(low_1d).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(high_1d).sub(pd.Series(low_1d))
    tr2 = pd.Series(high_1d).sub(pd.Series(close_1d.shift(1))).abs()
    tr3 = pd.Series(low_1d).sub(pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di_14 = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_14)
    minus_di_14 = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_14)
    dx = (abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)) * 100
    adx_14 = dx.ewm(alpha=1/14, adjust=False).mean()
    
    adx_14_values = adx_14.values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_values)
    
    # Calculate Donchian(20) on 12h timeframe for breakout confirmation
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50)  # warmup for volume MA and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_camarilla_r3 = camarilla_r3_aligned[i]
        curr_camarilla_s3 = camarilla_s3_aligned[i]
        curr_adx = adx_14_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and ADX > 25 (trending market)
            if curr_volume_spike and curr_adx > 25:
                # Bullish entry: break above Donchian high AND above Camarilla R3
                if curr_high > curr_donchian_high and curr_close > curr_camarilla_r3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below Donchian low AND below Camarilla S3
                elif curr_low < curr_donchian_low and curr_close < curr_camarilla_s3:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below Donchian low (breakout fails)
            if curr_close < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high (breakdown fails)
            if curr_close > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals