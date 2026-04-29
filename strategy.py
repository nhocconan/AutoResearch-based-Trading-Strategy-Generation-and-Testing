#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels (R1, S1) from weekly range act as strong support/resistance - breaks often lead to sustained moves
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Volume spike confirms breakout validity
# ATR-based stoploss manages risk
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
# Works in bull markets via trend-following breaks and in bear markets via mean-reversion at extremes

name = "1d_Camarilla_R1S1_Breakout_1wEMA50_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Need at least 1 previous bar to calculate Camarilla levels
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous bar's weekly range
        # Use weekly high/low from HTF data aligned to current bar
        idx_1w = (i // (7 * 24 * 4))  # approximate weekly bar index
        if idx_1w < 1 or idx_1w >= len(df_1w):
            signals[i] = 0.0
            continue
            
        # Get previous completed weekly bar for Camarilla calculation
        weekly_high = df_1w['high'].values[idx_1w-1]
        weekly_low = df_1w['low'].values[idx_1w-1]
        weekly_close = df_1w['close'].values[idx_1w-1]
        weekly_range = weekly_high - weekly_low
        
        if weekly_range <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels from weekly range
        R1 = weekly_close + weekly_range * 1.1 / 12
        S1 = weekly_close - weekly_range * 1.1 / 12
        
        curr_close = close[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below S1 OR price below 1w EMA50 OR stoploss hit
            if curr_close < S1 or curr_close < curr_ema_1w or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above R1 OR price above 1w EMA50 OR stoploss hit
            if curr_close > R1 or curr_close > curr_ema_1w or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R1 AND price > 1w EMA50 AND volume spike
            if curr_close > R1 and curr_close > curr_ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below S1 AND price < 1w EMA50 AND volume spike
            elif curr_close < S1 and curr_close < curr_ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals