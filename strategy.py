#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Uses 1w EMA34 for ultra-stable trend direction and requires volume > 1.5x average for confirmation.
# Designed for very low trade frequency (~12-37 trades/year) to minimize fee drag.
# Only takes breakouts in the direction of the 1w EMA34 trend, works in bull/bear markets.
# Added ATR-based stoploss (2.0x ATR) to manage risk and reduce whipsaws.

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR for stoploss (using 14-period ATR on 12h)
    if n >= 14:
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        atr = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla levels for previous 1w bar (completed)
        if len(df_1w) >= 2:
            # Camarilla levels: based on previous 1w bar's range
            prev_high = df_1w['high'].iloc[-2]
            prev_low = df_1w['low'].iloc[-2]
            prev_close = df_1w['close'].iloc[-2]
            
            # Calculate Camarilla levels
            range_val = prev_high - prev_low
            camarilla_r3 = prev_close + (range_val * 1.1 / 4)
            camarilla_s3 = prev_close - (range_val * 1.1 / 4)
            camarilla_r4 = prev_close + (range_val * 1.1 / 2)
            camarilla_s4 = prev_close - (range_val * 1.1 / 2)
            
            # Create arrays aligned to 1w timeframe
            camarilla_r3_1w = np.full(len(df_1w), camarilla_r3)
            camarilla_s3_1w = np.full(len(df_1w), camarilla_s3)
            camarilla_r4_1w = np.full(len(df_1w), camarilla_r4)
            camarilla_s4_1w = np.full(len(df_1w), camarilla_s4)
            
            # Align to 12h timeframe with proper delay (wait for 1w bar to close)
            camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
            camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
        else:
            camarilla_r3_aligned = np.full(n, np.nan)
            camarilla_s3_aligned = np.full(n, np.nan)
            camarilla_r4_aligned = np.full(n, np.nan)
            camarilla_s4_aligned = np.full(n, np.nan)
        
        # Volume confirmation: volume > 1.5x 20-period average (balanced)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, 1w EMA34 uptrend, volume spike
            if (curr_close > camarilla_r3_aligned[i] and 
                curr_close > curr_ema_34_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3, 1w EMA34 downtrend, volume spike
            elif (curr_close < camarilla_s3_aligned[i] and 
                  curr_close < curr_ema_34_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Camarilla S3, or ATR stoploss hit
            if curr_close < camarilla_s3_aligned[i] or curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Camarilla R3, or ATR stoploss hit
            if curr_close > camarilla_r3_aligned[i] or curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals