#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA200 trend filter and volume confirmation.
# Uses 4h/1d for signal direction (trend + structure), 1h only for entry timing precision.
# Long when price breaks above Camarilla R3 with 4h EMA200 uptrend and volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 with 4h EMA200 downtrend and volume confirmation.
# Discrete sizing 0.20. ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Session filter: 08-20 UTC to reduce noise trades.
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag.

name = "1h_Camarilla_R3S3_4hEMA200_Trend_VolumeSpike_v1"
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
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA200 trend filter
    ema_200 = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    # We need previous day's OHLC for today's levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (camarilla_range * 1.1 / 4)
    s3 = prev_close - (camarilla_range * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 200  # warmup for EMA200 and ATR
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema_200_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-bar average (tight to reduce trades)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        # Camarilla breakout conditions
        breakout_up = curr_close > r3_aligned[i-1]  # break above previous R3
        breakout_down = curr_close < s3_aligned[i-1]  # break below previous S3
        
        # Trend filter: bullish if close > EMA200, bearish if close < EMA200
        bullish_trend = curr_close > ema_200_aligned[i]
        bearish_trend = curr_close < ema_200_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up AND bullish trend AND volume confirmation
            if (breakout_up and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: Camarilla breakout down AND bearish trend AND volume confirmation
            elif (breakout_down and 
                  bearish_trend and 
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
            # Exit: price re-enters Camarilla range (between S3 and R3) OR trend turns bearish
            elif (curr_close < r3_aligned[i] and curr_close > s3_aligned[i]) or \
                 bearish_trend:
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
            # Exit: price re-enters Camarilla range (between S3 and R3) OR trend turns bullish
            elif (curr_close < r3_aligned[i] and curr_close > s3_aligned[i]) or \
                 bullish_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals