#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakout with 1d trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance where institutional order flow clusters.
# Breakouts above R1 or below S1 with volume spike indicate strong short-term participation.
# 1d EMA(34) ensures alignment with daily trend to avoid counter-trend trades.
# Session filter (08-20 UTC) reduces noise during low-liquidity periods.
# Designed for low trade frequency (15-37/year) to minimize fee drag in both bull and bear markets.

name = "1h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1"
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
    
    # Precompute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R1, S1, R2, S2)
    # Based on previous 4h bar's high, low, close
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_4h + low_4h + close_4h) / 3.0
    # Calculate Camarilla levels
    r1 = pp + (high_4h - low_4h) * 1.1 / 12.0
    s1 = pp - (high_4h - low_4h) * 1.1 / 12.0
    r2 = pp + (high_4h - low_4h) * 1.1 / 6.0
    s2 = pp - (high_4h - low_4h) * 1.1 / 6.0
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_4h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_4h, s2)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_r2 = r2_aligned[i]
        curr_s2 = s2_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and session
            if volume_spike:
                # Bullish entry: price breaks above 4h Camarilla R1 with 1d uptrend
                if curr_close > curr_r1 and curr_close > curr_ema:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h Camarilla S1 with 1d downtrend
                elif curr_close < curr_s1 and curr_close < curr_ema:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 4h Camarilla S1
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 4h Camarilla R2
            elif curr_close >= curr_r2:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 4h Camarilla R1
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 4h Camarilla S2
            elif curr_close <= curr_s2:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.20
    
    return signals