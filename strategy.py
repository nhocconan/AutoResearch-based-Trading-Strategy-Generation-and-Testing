#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume confirmation.
# Long when price breaks above Camarilla R3 (1h) with 1d volume > 1.5x 20-period average and price > 4h EMA50.
# Short when price breaks below Camarilla S3 (1h) with volume confirmation and price < 4h EMA50.
# Uses discrete sizing 0.20. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla levels calculated from prior completed 1h bar to avoid look-ahead.
# Volume spike filters low-momentum breakouts. 4h EMA50 ensures trades only in established trends.
# Works in bull (breakouts with strong uptrend) and bear (breakouts with strong downtrend) regimes.
# Target: 15-37 trades/year on 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate session hours (08-20 UTC) - precompute before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and Camarilla
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current 1d volume > 1.5x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 1.5)
        
        # Trend filter: price vs 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Calculate Camarilla levels for current 1h bar (using previous completed bar)
        # Need at least one completed 1h bar for calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Get the previous completed 1h bar
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla levels calculation
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_r3 = prev_close + (range_val * 1.1 / 4)
        camarilla_s3 = prev_close - (range_val * 1.1 / 4)
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla R3 breakout up AND volume spike AND uptrend AND in session
            if (curr_high > camarilla_r3 and 
                volume_spike and 
                uptrend):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: Camarilla S3 breakout down AND volume spike AND downtrend AND in session
            elif (curr_low < camarilla_s3 and 
                  volume_spike and 
                  downtrend):
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
            # Exit: price re-enters Camarilla H-L range OR trend reverses
            elif (curr_low >= camarilla_s3 and curr_high <= camarilla_r3) or \
                 (curr_close < ema_50_4h_aligned[i]):  # trend reversal
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
            # Exit: price re-enters Camarilla H-L range OR trend reverses
            elif (curr_high >= camarilla_s3 and curr_low <= camarilla_r3) or \
                 (curr_close > ema_50_4h_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals