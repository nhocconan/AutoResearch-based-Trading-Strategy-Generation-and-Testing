#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level with volume > 1.5x 20-period volume average and price > 12h EMA50 (uptrend).
# Short when price breaks below Camarilla S3 level with volume confirmation and price < 12h EMA50 (downtrend).
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla levels calculated from prior completed 4h bar to avoid look-ahead.
# Volume confirmation filters low-momentum breakouts. 12h EMA50 ensures trades only in established trends.
# Works in bull (breakouts with strong uptrend) and bear (breakouts with strong downtrend) regimes.
# Target: 30-60 trades/year on 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 12h data ONCE before loop for EMA and volume filters (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and Camarilla
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 12h volume average
        if vol_ma_12h_aligned[i] <= 0 or np.isnan(vol_ma_12h_aligned[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_12h_aligned[i] * 1.5)
        
        # Trend filter: price vs 12h EMA50
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Load 4h data ONCE before loop for Camarilla levels
        df_4h = get_htf_data(prices, '4h')
        if len(df_4h) < 5:
            signals[i] = 0.0
            continue
        
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        close_4h = df_4h['close'].values
        
        # Calculate Camarilla levels for each 4h bar (using previous completed bar)
        # Camarilla R3 = close + 1.1*(high-low)/2
        # Camarilla S3 = close - 1.1*(high-low)/2
        # Use previous completed bar to avoid look-ahead
        prev_high = high_4h[:-1]
        prev_low = low_4h[:-1]
        prev_close = close_4h[:-1]
        
        # Need at least 1 previous bar
        if len(prev_high) < 1:
            signals[i] = 0.0
            continue
            
        # Calculate levels for previous bar
        camarilla_range = prev_high - prev_low
        r3 = prev_close + 1.1 * camarilla_range / 2
        s3 = prev_close - 1.1 * camarilla_range / 2
        
        # Shift to align with current bar (use previous bar's levels)
        r3_aligned = np.concatenate([[np.nan], r3])
        s3_aligned = np.concatenate([[np.nan], s3])
        
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = curr_high > r3_level  # break above R3
        breakout_down = curr_low < s3_level  # break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up AND volume confirmation AND uptrend
            if (breakout_up and 
                volume_confirm and 
                uptrend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Camarilla breakout down AND volume confirmation AND downtrend
            elif (breakout_down and 
                  volume_confirm and 
                  downtrend):
                signals[i] = -0.25
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
            # Exit: price re-enters Camarilla bands OR trend reverses
            elif (curr_low >= s3_level and curr_low <= r3_level) or \
                 (curr_close < ema_50_12h_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla bands OR trend reverses
            elif (curr_high >= s3_level and curr_high <= r3_level) or \
                 (curr_close > ema_50_12h_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals