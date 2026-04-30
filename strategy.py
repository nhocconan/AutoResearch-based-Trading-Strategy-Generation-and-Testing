#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 levels with volume confirmation and 1d trend filter
# Camarilla pivots identify key intraday support/resistance where institutional order flow clusters.
# Breakouts above R3 or below S3 with volume spike indicate strong institutional participation.
# 1d EMA(50) ensures alignment with longer-term trend to avoid counter-trend trades.
# Session filter (08-20 UTC) reduces noise trades outside active market hours.
# Designed for moderate trade frequency (~30-60/year) to balance opportunity and fee drag on 1h timeframe.
# Uses 1h timeframe as primary, with 4h for signal direction and 1d for trend filter.

name = "1h_Camarilla_R3S3_Breakout_4hDir_1dTrend_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla calculation (signal direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R3, S3, R4, S4)
    # Based on previous 4h bar's high, low, close
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_4h + low_4h + close_4h) / 3.0
    # Calculate Camarilla levels
    r3 = pp + (high_4h - low_4h) * 1.1 / 4.0
    s3 = pp - (high_4h - low_4h) * 1.1 / 4.0
    r4 = pp + (high_4h - low_4h) * 1.1 / 2.0
    s4 = pp - (high_4h - low_4h) * 1.1 / 2.0
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_4h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_4h, s4)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position == 1:
                # Close long at session end if still in session boundary
                signals[i] = 0.0
                position = 0
            elif position == -1:
                # Close short at session end if still in session boundary
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike, session, and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 4h Camarilla R3 with 1d uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h Camarilla S3 with 1d downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 4h Camarilla S3
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 4h Camarilla R4
            elif curr_close >= curr_r4:
                signals[i] = 0.0  # full exit
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 4h Camarilla R3
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 4h Camarilla S4
            elif curr_close <= curr_s4:
                signals[i] = 0.0  # full exit
                position = 0
            else:
                signals[i] = -0.20
    
    return signals