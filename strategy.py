#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike.
# Long when price breaks above Camarilla R3 with 4h EMA50 uptrend and volume > 2x 20-bar average.
# Short when price breaks below Camarilla S3 with 4h EMA50 downtrend and volume confirmation.
# Uses discrete sizing 0.20. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Session filter: 08-20 UTC to reduce noise trades. Target: 15-30 trades/year to minimize fee drag on 1h.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla levels (R3, S3) based on previous day
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using daily OHLC from 1d timeframe for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = 1.1 * (high_1d - low_1d)
    r3_1d = close_1d + camarilla_range * 1.1 / 4
    s3_1d = close_1d - camarilla_range * 1.1 / 4
    
    # Align daily Camarilla levels to 1h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation: current volume > 2x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(atr[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2x 20-bar average
        volume_confirm = curr_volume > (vol_ma[i] * 2.0) if vol_ma[i] > 0 else False
        
        # 4h EMA50 trend filter: uptrend if close > EMA50, downtrend if close < EMA50
        ema_trend_up = curr_close > ema_50_4h_aligned[i]
        ema_trend_down = curr_close < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R3 with uptrend and volume confirmation
            if (curr_high > r3_1d_aligned[i] and 
                ema_trend_up and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: break below S3 with downtrend and volume confirmation
            elif (curr_low < s3_1d_aligned[i] and 
                  ema_trend_down and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range (between S3 and R3) OR trend reverses
            elif (curr_low >= s3_1d_aligned[i] and curr_high <= r3_1d_aligned[i]) or \
                 (ema_trend_down and not ema_trend_up):  # trend changed to down
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range OR trend reverses
            elif (curr_low >= s3_1d_aligned[i] and curr_high <= r3_1d_aligned[i]) or \
                 (ema_trend_up and not ema_trend_down):  # trend changed to up
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals