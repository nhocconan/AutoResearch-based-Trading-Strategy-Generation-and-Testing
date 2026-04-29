#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA200 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13 to detect trend strength
# Uses 1d EMA200 for higher timeframe trend alignment (works in both bull/bear markets)
# Volume confirmation > 1.8x average to filter weak signals
# Discrete position sizing (0.25) with ATR-based stop loss via signal=0
# Designed to capture strong trends while avoiding whipsaws in ranging markets

name = "6h_ElderRay_1dEMA200_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate ATR(14) for stop loss and Elder Ray smoothing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA13 for Elder Ray (using close prices)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 200)  # Warmup for EMA200 and ATR
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_ema200_1d = ema_200_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate 20-period average volume for confirmation
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = np.nan
        
        if np.isnan(vol_ma_20):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirmed = curr_volume > 1.8 * vol_ma_20
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stop loss (2.0 * ATR below entry) or bear power turns negative
            stop_loss = entry_price - 2.0 * curr_atr
            if curr_low <= stop_loss or curr_bear > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stop loss (2.0 * ATR above entry) or bull power turns positive
            stop_loss = entry_price + 2.0 * curr_atr
            if curr_high >= stop_loss or curr_bull < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation
            if not vol_confirmed:
                signals[i] = 0.0
                continue
            
            # Long when bull power > 0 (strong buying), price above 1d EMA200 (uptrend), volume confirmed
            if curr_bull > 0 and curr_close > curr_ema200_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when bear power < 0 (strong selling), price below 1d EMA200 (downtrend), volume confirmed
            elif curr_bear < 0 and curr_close < curr_ema200_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals