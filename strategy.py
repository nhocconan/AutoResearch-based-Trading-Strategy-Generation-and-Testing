#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) from 1d for high-probability reversal/breakout zones
# 1d EMA34 provides trend filter to align with higher timeframe momentum
# Volume spike (2.0x 20-period average) confirms breakout validity
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag
# Works in bull markets via trend-following breaks and avoids counter-trend trades in bear markets

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA
    
    for i in range(start_idx, n):
        # Need at least 5 previous bars for Camarilla calculation
        if i < 5:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous 1d bar (using previous completed 1d bar)
        # We use the previous day's high, low, close to calculate today's levels
        # Since we're on 6h timeframe, we need to get the 1d OHLC from the HTF data
        # Get the index of the 1d bar that corresponds to the current 6h bar
        # We'll use the previous completed 1d bar for Camarilla calculation
        if i >= 4:  # Need at least 4 6h bars to have completed 1d (4*6h=24h)
            # Find the 1d bar index that completed before current 6h bar
            # Each 1d bar = 4 6h bars
            idx_1d = (i // 4) - 1  # Previous completed 1d bar
            if idx_1d >= 0 and idx_1d < len(df_1d):
                prev_high = df_1d['high'].iloc[idx_1d]
                prev_low = df_1d['low'].iloc[idx_1d]
                prev_close = df_1d['close'].iloc[idx_1d]
                
                # Calculate Camarilla levels
                range_val = prev_high - prev_low
                camarilla_r3 = prev_close + range_val * 1.1 / 4
                camarilla_s3 = prev_close - range_val * 1.1 / 4
                camarilla_r4 = prev_close + range_val * 1.1 / 2
                camarilla_s4 = prev_close - range_val * 1.1 / 2
            else:
                camarilla_r3 = camarilla_s3 = camarilla_r4 = camarilla_s4 = 0.0
        else:
            camarilla_r3 = camarilla_s3 = camarilla_r4 = camarilla_s4 = 0.0
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 1.5 * ATR below entry
            stop_price = entry_price - 1.5 * curr_atr
            # Exit conditions: price below Camarilla S3 OR price below 1d EMA34 OR stoploss hit
            if curr_close < camarilla_s3 or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 1.5 * ATR above entry
            stop_price = entry_price + 1.5 * curr_atr
            # Exit conditions: price above Camarilla R3 OR price above 1d EMA34 OR stoploss hit
            if curr_close > camarilla_r3 or curr_close > curr_ema_1d or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if curr_high > camarilla_r3 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif curr_low < camarilla_s3 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals