#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla breakouts with volume confirmation and session filter.
# Long when price breaks above 4h Camarilla R3 AND 1d EMA34 uptrend AND volume > 1.5x 20-bar median AND 08-20 UTC.
# Short when price breaks below 4h Camarilla S3 AND 1d EMA34 downtrend AND volume > 1.5x 20-bar median AND 08-20 UTC.
# Uses discrete sizing 0.20. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# 4h Camarilla provides structure; 1d EMA34 filters trend; volume spike confirms conviction; session filter reduces noise.
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years).

name = "1h_Camarilla_R3S3_Breakout_4h_1dEMA34_Volume_Session_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 4h Camarilla levels (HTF) - using prior bar's OHLC to avoid look-ahead
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    prev_close_4h = np.concatenate([[df_4h['close'].values[0]], df_4h['close'].values[:-1]])
    prev_high_4h = np.concatenate([[df_4h['high'].values[0]], df_4h['high'].values[:-1]])
    prev_low_4h = np.concatenate([[df_4h['low'].values[0]], df_4h['low'].values[:-1]])
    camarilla_range_4h = prev_high_4h - prev_low_4h
    camarilla_R3_4h = prev_close_4h + 1.125 * camarilla_range_4h
    camarilla_S3_4h = prev_close_4h - 1.125 * camarilla_range_4h
    camarilla_R3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R3_4h)
    camarilla_S3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S3_4h)
    
    # Calculate 1d EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, volume, and Camarilla
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_R3_4h_aligned[i]) or 
            np.isnan(camarilla_S3_4h_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > 4h Camarilla R3 AND uptrend AND volume spike
            if curr_close > camarilla_R3_4h_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price < 4h Camarilla S3 AND downtrend AND volume spike
            elif curr_close < camarilla_S3_4h_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below 4h Camarilla S3 OR trend turns down
            elif curr_close < camarilla_S3_4h_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above 4h Camarilla R3 OR trend turns up
            elif curr_close > camarilla_R3_4h_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals