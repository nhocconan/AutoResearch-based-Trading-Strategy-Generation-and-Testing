#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation.
# Camarilla pivot levels (R3, S3) act as strong support/resistance. Breakout above R3 or below S3
# with 1w EMA50 trend alignment and volume > 2x 20-bar average triggers entry.
# Uses discrete sizing 0.30 to balance return and drawdown. Designed for 12h timeframe
# to capture multi-day swings while minimizing fee churn. Works in bull (buy R3 breakout in uptrend)
# and bear (sell S3 breakdown in downtrend) via 1w trend filter.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d (based on previous day's OHLC)
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    # We focus on R3 and S3 as key breakout levels
    
    # Shift OHLC by 1 to use previous day's data for today's levels
    prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prev_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    # Calculate Camarilla R3 and S3
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        # Volume confirmation: current 12h volume > 2.0x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 AND 1w EMA50 uptrend (price > EMA50) AND volume confirmation
            if (curr_high > curr_r3 and 
                curr_close > curr_ema_50_1w and 
                volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla S3 AND 1w EMA50 downtrend (price < EMA50) AND volume confirmation
            elif (curr_low < curr_s3 and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirm):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below 1w EMA50 (trend change) OR price retests Camarilla PP (profit target)
            # Calculate Camarilla PP for exit
            camarilla_pp = (prev_high[i] + prev_low[i] + prev_close[i]) / 3 if not (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i])) else np.nan
            camarilla_pp_aligned = camarilla_pp  # PP is already aligned via index i
            if np.isnan(camarilla_pp_aligned):
                camarilla_pp_aligned = curr_close  # fallback
            
            if (curr_close < curr_ema_50_1w or 
                curr_close < camarilla_pp_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price crosses above 1w EMA50 (trend change) OR price retests Camarilla PP (profit target)
            camarilla_pp = (prev_high[i] + prev_low[i] + prev_close[i]) / 3 if not (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i])) else np.nan
            camarilla_pp_aligned = camarilla_pp  # PP is already aligned via index i
            if np.isnan(camarilla_pp_aligned):
                camarilla_pp_aligned = curr_close  # fallback
            
            if (curr_close > curr_ema_50_1w or 
                curr_close > camarilla_pp_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals