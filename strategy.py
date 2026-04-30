#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h ADX trend filter and volume confirmation
# Camarilla pivots from 1d provide intraday support/resistance levels. R3/S3 are strong reversal zones;
# breakouts beyond R4/S4 indicate continuation. 12h ADX > 25 filters for trending markets only.
# Volume spike confirms breakout validity. Works in bull via breakout longs, in bear via breakout shorts.
# Discrete sizing 0.25 balances risk and minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R3S3_Breakout_12hADX25_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    # Use previous day's OHLC for today's Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r4 = pivot + (prev_high - prev_low) * 1.1 / 2.0
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    s4 = pivot - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 12h ADX(14) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    # True Range
    tr1 = pd.Series(df_12h['high']).shift(1) - pd.Series(df_12h['low']).shift(1)
    tr2 = abs(pd.Series(df_12h['high']).shift(1) - pd.Series(df_12h['close']).shift(1))
    tr3 = abs(pd.Series(df_12h['low']).shift(1) - pd.Series(df_12h['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Directional Movement
    up_move = pd.Series(df_12h['high']) - pd.Series(df_12h['high']).shift(1)
    down_move = pd.Series(df_12h['low']).shift(1) - pd.Series(df_12h['low'])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_14
    dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14)  # warmup for volume MA and 12h ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r4 = r4_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market (ADX > 25)
            if curr_volume_spike and curr_adx > 25:
                # Bullish breakout: price breaks above R4
                if curr_close > curr_r4:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below S4
                elif curr_close < curr_s4:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R3 (mean reversion) or breaks below S4 (stop)
            if curr_close < curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above S3 (mean reversion) or breaks above R4 (stop)
            if curr_close > curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals