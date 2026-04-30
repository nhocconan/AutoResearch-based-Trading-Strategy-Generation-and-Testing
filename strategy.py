#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme + 1d EMA50 trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions; extremes (<-90 or >-10) signal potential reversals.
# 1d EMA50 ensures trades align with long-term trend to avoid counter-trend whipsaws.
# Volume > 1.8x 24-period average confirms momentum (moderate threshold to balance signal quality and frequency).
# ATR-based stoploss (2.0x) manages risk. Target: 15-25 trades/year to minimize fee drag on 12h timeframe.

name = "12h_WilliamsR_Extreme_1dEMA50_VolumeConfirm_ATRStop_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d data
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for 12h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams %R(14) for 12h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close) / (highest_high - lowest_low),
                          -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for EMA, ATR, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(williams_r[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_aligned[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r[i]
        
        # Volume confirmation: volume > 1.8x 24-period average (moderate threshold)
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-24:i])
            volume_confirm = volume[i] > (1.8 * vol_ma_24)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -90 (oversold), price above 1d EMA50, volume confirmation
            if (curr_williams_r < -90 and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R > -10 (overbought), price below 1d EMA50, volume confirmation
            elif (curr_williams_r > -10 and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: Williams %R > -50 (exit oversold) OR stoploss hit
            if (curr_williams_r > -50 or 
                curr_close < entry_price - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Williams %R < -50 (exit overbought) OR stoploss hit
            if (curr_williams_r < -50 or 
                curr_close > entry_price + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals