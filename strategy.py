#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot (R3/S3) breakout with 1d trend filter and volume confirmation
# Camarilla pivots identify institutional support/resistance levels; breakouts with volume confirm institutional participation;
# 1d EMA(34) ensures alignment with long-term trend. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (15-37/year) to minimize fee drag in both bull and bear markets.

name = "1h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "1h"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # Camarilla uses previous day's range to calculate support/resistance
    df_4h['date'] = df_4h.index.date
    daily_ohlc = df_4h.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily_ohlc) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_prev = daily_ohlc['high'].shift(1).values
    low_prev = daily_ohlc['low'].shift(1).values
    close_prev = daily_ohlc['close'].shift(1).values
    
    # Camarilla R3, S3 levels
    camarilla_r3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    camarilla_s3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Map daily levels to 4h bars (each 4h bar gets the Camarilla levels of its day)
    camarilla_r3_4h = np.repeat(camarilla_r3, 6)[:len(df_4h)]  # 6x 4h bars per day
    camarilla_s3_4h = np.repeat(camarilla_s3, 6)[:len(df_4h)]
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Calculate 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 2.0x 24-period average (1 day)
    vol_ma_24 = np.zeros(n)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Require volume spike and 1d trend alignment
        if not volume_spike[i]:
            signals[i] = 0.0 if position == 0 else signals[i-1]
            continue
            
        curr_close = close[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish entry: price breaks above Camarilla R3 with 1d uptrend
            if curr_close > curr_r3 and curr_close > curr_ema:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Bearish entry: price breaks below Camarilla S3 with 1d downtrend
            elif curr_close < curr_s3 and curr_close < curr_ema:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks Camarilla S3
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla R3 (full retracement)
            elif curr_close >= curr_r3:
                signals[i] = 0.0  # exit full position
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks Camarilla R3
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla S3 (full retracement)
            elif curr_close <= curr_s3:
                signals[i] = 0.0  # exit full position
            else:
                signals[i] = -0.20
    
    return signals