#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with volume spike and 12h EMA50 trend filter.
# Long when price breaks above Camarilla R3 with volume > 2.0x 20-bar average and close > 12h EMA50.
# Short when price breaks below Camarilla S3 with volume confirmation and close < 12h EMA50.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Target: 20-40 trades/year to minimize fee drag and avoid overtrading.
# Camarilla levels provide strong intraday support/resistance; volume confirms institutional interest;
# 12h EMA50 ensures alignment with medium-term trend to reduce whipsaws in ranging markets.

name = "4h_Camarilla_R3S3_Volume_12hEMA50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # Typical Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #                    S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use daily high/low/close to compute levels for current 4h session
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.125 * camarilla_range
    s3 = prev_close - 1.125 * camarilla_range
    
    # Align 1d Camarilla levels to 4h (they represent levels for the current day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 20  # warmup for volume average
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(atr[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above R3 AND volume confirmation AND price above 12h EMA50 (uptrend)
            if (curr_high > r3_aligned[i] and 
                volume_confirm and 
                curr_close > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Break below S3 AND volume confirmation AND price below 12h EMA50 (downtrend)
            elif (curr_low < s3_aligned[i] and 
                  volume_confirm and 
                  curr_close < ema_50_12h_aligned[i]):
                signals[i] = -0.25
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
            elif (curr_low >= s3_aligned[i] and curr_low <= r3_aligned[i]) or \
                 (curr_close < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range OR trend reverses
            elif (curr_high >= s3_aligned[i] and curr_high <= r3_aligned[i]) or \
                 (curr_close > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals