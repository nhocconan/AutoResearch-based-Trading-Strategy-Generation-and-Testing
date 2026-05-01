#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 AND close > 4h EMA50 AND volume > 1.5x 1h volume median.
# Short when price breaks below S3 AND close < 4h EMA50 AND volume > 1.5x 1h volume median.
# Uses discrete sizing 0.20. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Session filter: only trade 08-20 UTC to reduce noise.
# Target: 15-37 trades/year (60-150 total over 4 years) on 1h timeframe to minimize fee drag.
# Works in bull markets via breakout entries and in bear markets via short alignments with volume confirmation.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_Session_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h volume median (20-period for stability)
    vol_median_1h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 4h EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate prior day's high, low, close for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Extract prior day's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    camarilla_range = prior_high - prior_low
    r3 = prior_close + camarilla_range * 1.1 / 2
    s3 = prior_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_median_1h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 1h volume median
        if vol_median_1h[i] <= 0 or np.isnan(vol_median_1h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_1h[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND uptrend AND volume spike
            if curr_close > r3_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below S3 AND downtrend AND volume spike
            elif curr_close < s3_aligned[i] and downtrend and volume_confirm:
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
            # Exit: price breaks below S3 OR trend turns down
            elif curr_close < s3_aligned[i] or not uptrend:
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
            # Exit: price breaks above R3 OR trend turns up
            elif curr_close > r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals