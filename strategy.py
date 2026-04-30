#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses 4h HTF for Camarilla pivot levels to reduce noise and 1d EMA50 for strong trend alignment.
# Long when price breaks above 4h R3 in uptrend (1d close > 1d EMA50) with volume spike (>2.0x average).
# Short when price breaks below 4h S3 in downtrend (1d close < 1d EMA50) with volume spike.
# Designed for moderate trade frequency (~30-60/year on 1h) to balance opportunity and fee drag.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods and reduce false signals.
# Stoploss via signal=0 when price reverses below/above opposite Camarilla level (S3/R3).
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at 4h pivot levels.

name = "1h_4hCamarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R3, S3) using typical price
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_s3 = close_4h - 1.1 * (high_4h - low_4h) / 2
    
    # Align 4h Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(14) for volatility filter (optional, not used in entry but for context)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else signals[i-1]  # hold position if already in trade
            continue
        
        # Volume confirmation: volume > 2.0x 50-period average
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
        elif i > 0:
            vol_ma_50 = np.mean(volume[:i])
        else:
            vol_ma_50 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 4h R3 with 1d uptrend (close > EMA50)
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h S3 with 1d downtrend (close < EMA50)
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: price breaks below 4h S3 (reversal signal) OR reverse signal
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Optional: exit on reverse bearish signal with volume
            elif curr_close < curr_ema and volume_spike and curr_close < curr_s3 * 1.001:  # near S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above 4h R3 (reversal signal) OR reverse signal
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Optional: exit on reverse bullish signal with volume
            elif curr_close > curr_ema and volume_spike and curr_close > curr_r3 * 0.999:  # near R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals