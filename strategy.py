#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversion with 12h EMA34 trend filter and volume confirmation.
# Uses 12h EMA34 for trend direction to avoid counter-trend trades in strong moves.
# Williams %R < -80 for long entry, > -20 for short entry (extreme oversold/overbought).
# Volume > 2.0x 24-period average confirms momentum (high threshold to reduce trade frequency).
# ATR-based stoploss (2.0x) limits drawdown. Designed for low trade frequency (~15-25 trades/year)
# to minimize fee drag on 6h timeframe. Works in bull/bear via 12h EMA34 trend filter +
# extreme mean reversion + volume confirmation.

name = "6h_WilliamsR_Extreme_12hEMA34_VolumeConfirm_ATRStop_v1"
timeframe = "6h"
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
    
    # Load 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h data
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Williams %R(14) for 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close) / (highest_high - lowest_low),
                          -50)  # neutral when range=0
    
    # Calculate ATR(14) for 6h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_aligned[i]
        curr_wr = williams_r[i]
        curr_atr = atr[i]
        
        # Volume confirmation: volume > 2.0x 24-period average (high threshold to reduce trades)
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-24:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_24)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (extreme oversold), price above 12h EMA34, volume spike
            if (curr_wr < -80 and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R > -20 (extreme overbought), price below 12h EMA34, volume spike
            elif (curr_wr > -20 and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: Williams %R > -50 (mean reversion) OR stoploss hit
            if (curr_wr > -50 or 
                curr_close < entry_price - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Williams %R < -50 (mean reversion) OR stoploss hit
            if (curr_wr < -50 or 
                curr_close > entry_price + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals