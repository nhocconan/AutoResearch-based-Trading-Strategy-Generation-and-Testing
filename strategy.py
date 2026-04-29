#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel (20-period high/low) from 12h data for breakout signals
# 1w EMA50 provides strong trend filter to avoid counter-trend trades in both bull and bear markets
# Volume spike (2.0x 20-period average) confirms breakout validity
# ATR-based stoploss (1.5x ATR) manages risk
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag
# Works in bull markets via upper band breakouts and in bear markets via lower band breakdowns with trend alignment

name = "12h_Donchian_Breakout_1wEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 12h data
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i-donchian_period+1:i+1])
        lower[i] = np.min(low[i-donchian_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, donchian_period - 1)  # warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper[i]
        curr_lower = lower[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 1.5 * ATR below entry
            stop_price = entry_price - 1.5 * curr_atr
            # Exit conditions: price below lower band OR price below 1w EMA50 OR stoploss hit
            if curr_close < curr_lower or curr_close < curr_ema_1w or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 1.5 * ATR above entry
            stop_price = entry_price + 1.5 * curr_atr
            # Exit conditions: price above upper band OR price above 1w EMA50 OR stoploss hit
            if curr_close > curr_upper or curr_close > curr_ema_1w or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above upper band AND price > 1w EMA50 AND volume spike
            if curr_high > curr_upper and curr_close > curr_ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below lower band AND price < 1w EMA50 AND volume spike
            elif curr_low < curr_lower and curr_close < curr_ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals