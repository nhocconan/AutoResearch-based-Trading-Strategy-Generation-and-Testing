#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1w trend filter + volume spike.
# Long when price breaks above Camarilla R3 with volume > 2.0x 24-bar average and 1w close > 1w EMA34.
# Short when price breaks below Camarilla S3 with volume confirmation and 1w close < 1w EMA34.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Target: 12-37 trades/year to minimize fee drag. Works in bull/bear via 1w trend filter.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels (based on previous bar's OHLC)
    # R3 = close + 1.1*(high - low)/2, S3 = close - 1.1*(high - low)/2
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 12h
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 1  # warmup for Camarilla (need previous bar)
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(atr[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 24-bar average
        vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND volume confirmation AND 1w bullish trend
            if (curr_high > r3[i] and 
                volume_confirm and 
                curr_close > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below S3 AND volume confirmation AND 1w bearish trend
            elif (curr_low < s3[i] and 
                  volume_confirm and 
                  curr_close < ema_34_1w_aligned[i]):
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
            # Exit: price re-enters Camarilla range (between R3 and S3) OR 1w trend turns bearish
            elif (curr_low <= r3[i] and curr_low >= s3[i]) or \
                 curr_close < ema_34_1w_aligned[i]:
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
            # Exit: price re-enters Camarilla range OR 1w trend turns bullish
            elif (curr_high <= r3[i] and curr_high >= s3[i]) or \
                 curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals