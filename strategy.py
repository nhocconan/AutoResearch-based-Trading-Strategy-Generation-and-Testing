#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume confirmation.
# Long when price breaks above Camarilla R1 AND close > 4h EMA50 AND volume > 2.0x 1d volume median.
# Short when price breaks below Camarilla S1 AND close < 4h EMA50 AND volume > 2.0x 1d volume median.
# Uses discrete sizing 0.20. ATR(10) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla levels from prior 1h provide structure; 4h EMA50 filters intermediate-term trend.
# Volume confirmation from 1d ensures institutional participation. Target: 15-25 trades/year on 1h timeframe.
# Session filter (08-20 UTC) avoids low-liquidity hours. Works in bull via breakouts, bear via mean reversion at extremes.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_1dVolume_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - avoids TypeError with datetime64
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(10) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1h volume median (24-period for stability)
    vol_median_1h = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    
    # Calculate 4h EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume median (for confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 24:
        return np.zeros(n)
    
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=24, min_periods=24).median().values
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    # Calculate Camarilla levels from prior 1h bar
    if len(df_4h) < 2:  # Need at least 2 completed 4h bars for prior 1h reference
        return np.zeros(n)
    
    # Resample 1h OHLC from 4h data is not allowed, so we use 4h close as proxy for prior period
    # Better approach: use 1h data directly for Camarilla (prior 1h bar)
    h1 = high[:-1]  # prior period high
    l1 = low[:-1]   # prior period low
    c1 = close[:-1] # prior period close
    
    # Shift to align with current bar (index i uses prior bar i-1)
    h1 = np.concatenate([[np.nan], h1])
    l1 = np.concatenate([[np.nan], l1])
    c1 = np.concatenate([[np.nan], c1])
    
    # Calculate Camarilla R1 and S1 levels
    # R1 = c1 + (h1 - l1) * 1.1/12
    # S1 = c1 - (h1 - l1) * 1.1/12
    camarilla_range = h1 - l1
    r1 = c1 + camarilla_range * 1.1 / 12.0
    s1 = c1 - camarilla_range * 1.1 / 12.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, volume, and Camarilla
    start_idx = 100
    
    for i in range(start_idx, n):
        if (not in_session[i] or 
            np.isnan(atr[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1[i]) or 
            np.isnan(s1[i]) or 
            np.isnan(vol_median_1h[i]) or 
            np.isnan(vol_median_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 1h volume median AND > 1.5x 1d volume median
        if vol_median_1h[i] <= 0 or vol_median_1d_aligned[i] <= 0:
            volume_confirm = False
        else:
            volume_confirm = (curr_volume > (vol_median_1h[i] * 2.0)) and (curr_volume > (vol_median_1d_aligned[i] * 1.5))
        
        if position == 0:  # Flat - look for new entries
            # Long: price > R1 AND uptrend AND volume spike
            if curr_close > r1[i] and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price < S1 AND downtrend AND volume spike
            elif curr_close < s1[i] and downtrend and volume_confirm:
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
            # Exit: price breaks below S1 OR trend turns down
            elif curr_close < s1[i] or not uptrend:
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
            # Exit: price breaks above R1 OR trend turns up
            elif curr_close > r1[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals