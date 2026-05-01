#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND price > 4h EMA50 AND volume > 1.5x 1h volume median.
# Short when price breaks below Camarilla S3 AND price < 4h EMA50 AND volume > 1.5x 1h volume median.
# Uses discrete sizing 0.20. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Session filter: only trade 08-20 UTC to avoid low-liquidity periods.
# Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years) to minimize fee drag.
# Camarilla pivots provide institutional structure, 4h EMA50 offers smooth trend filter, volume confirms momentum.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_Session"
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
    
    # Calculate 1h Camarilla pivot levels (R3, S3) from previous 1h bar
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous 1h bar's OHLC
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    prev_close = df_1h['close'].values
    prev_high = df_1h['high'].values
    prev_low = df_1h['low'].values
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (already aligned to previous bar close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    
    # Calculate 4h EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h volume median (20-period for stability)
    vol_median_1h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_median_1h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 1h volume median
        if vol_median_1h[i] <= 0 or np.isnan(vol_median_1h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_1h[i] * 1.5)
        
        # Trend filter: price vs 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Camarilla R3 AND uptrend AND volume confirmation
            if (curr_high > camarilla_r3_aligned[i] and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: Break below Camarilla S3 AND downtrend AND volume confirmation
            elif (curr_low < camarilla_s3_aligned[i] and 
                  downtrend and 
                  volume_confirm):
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
            # Exit: price breaks below Camarilla S3 OR trend turns down
            elif (curr_low < camarilla_s3_aligned[i]) or (not uptrend):
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
            # Exit: price breaks above Camarilla R3 OR trend turns up
            elif (curr_high > camarilla_r3_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals