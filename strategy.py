#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike.
# Long when price breaks above R3 with volume > 1.8x 20-bar average and 12h EMA50 rising.
# Short when price breaks below S3 with volume confirmation and 12h EMA50 falling.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla levels derived from prior 12h bar (HLC of previous completed 12h bar).
# Volume spike filters low-momentum breakouts. EMA50 trend ensures alignment with intermediate trend.
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years).
# Works in bull (breakouts with volume + trend) and bear (failed breaks reverse to opposite side).

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 12h data ONCE before loop for Camarilla levels and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar (using HLC of that bar)
    # Camarilla R3 = close + 1.1*(high - low)/2
    # Camarilla S3 = close - 1.1*(high - low)/2
    hl_range_12h = high_12h - low_12h
    r3_12h = close_12h + 1.1 * hl_range_12h / 2.0
    s3_12h = close_12h - 1.1 * hl_range_12h / 2.0
    
    # Calculate EMA50 on 12h close
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 12h indicators to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        if vol_ma[i] <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma[i] * 1.8)
        
        # Get current 12h levels (already aligned to 6h bars)
        r3 = r3_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        ema_50 = ema_50_12h_aligned[i]
        
        if np.isnan(r3) or np.isnan(s3) or np.isnan(ema_50):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = curr_high > r3  # break above R3
        breakout_down = curr_low < s3  # break below S3
        
        # EMA50 trend filter: rising for long, falling for short
        # Need previous EMA value to determine slope
        if i > start_idx:
            ema_50_prev = ema_50_12h_aligned[i-1]
            ema_rising = ema_50 > ema_50_prev
            ema_falling = ema_50 < ema_50_prev
        else:
            ema_rising = True  # default to allow first bar
            ema_falling = True
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND volume confirmation AND EMA50 rising
            if (breakout_up and 
                volume_confirm and 
                ema_rising):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: breakout below S3 AND volume confirmation AND EMA50 falling
            elif (breakout_down and 
                  volume_confirm and 
                  ema_falling):
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
            # Exit: price re-enters below R3 (failed breakout) OR EMA50 turns down
            elif (curr_close < r3) or (not ema_rising):
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
            # Exit: price re-enters above S3 (failed breakout) OR EMA50 turns up
            elif (curr_close > s3) or (not ema_falling):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals