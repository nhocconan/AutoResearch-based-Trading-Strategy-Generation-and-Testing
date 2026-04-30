#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Uses 4h/1d for signal direction, 1h only for entry timing precision. Targets 15-37 trades/year to minimize fee drag.
# Session filter (08-20 UTC) reduces noise trades. Position size 0.20 for balanced risk/return.

name = "1h_4hCamarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on prior 4h bar's OHLC)
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    hl_range_4h = df_4h['high'] - df_4h['low']
    camarilla_r3_4h = typical_price_4h + hl_range_4h * 1.1 / 4
    camarilla_s3_4h = typical_price_4h - hl_range_4h * 1.1 / 4
    
    # Align 4h Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h.values)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h.values)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 35  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_4h_aligned[i]
        curr_s3 = camarilla_s3_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 4h R3 with 1d uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h S3 with 1d downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 4h S3 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 4h R3 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals