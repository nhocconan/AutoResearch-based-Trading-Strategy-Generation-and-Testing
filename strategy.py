#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla pivot levels (R1/S1) with 4h EMA21 trend filter and volume spike confirmation
# Uses 4h HTF for Camarilla pivot calculation and 4h EMA21 for trend to filter false breakouts.
# Long when price breaks above 4h R1 in uptrend (1h close > 4h EMA21) with volume spike (>1.8x average).
# Short when price breaks below 4h S1 in downtrend (1h close < 4h EMA21) with volume spike.
# Uses 1d EMA50 as regime filter: only trade long when 1h close > 1d EMA50, short when 1h close < 1d EMA50.
# Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
# Designed for low trade frequency (~15-37/year on 1h) to minimize fee drag while capturing strong directional moves.
# Uses volume confirmation with moderate threshold (>1.8x average) to balance signal quality and trade count.
# Stoploss via signal=0 when price breaks opposite Camarilla level (S1 for long, R1 for short).
# Take profit at 1.5x ATR from entry to allow for swings while locking in gains.

name = "1h_4hCamarilla_R1S1_Breakout_4hEMA21_1dEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours for UTC 08-20 filter
    hours = prices.index.hour
    
    # Load 4h data ONCE before loop for Camarilla and EMA calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R1, S1) using typical price
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 12
    
    # Align 4h Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 4h EMA(21) for trend filter
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Load 1d data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for regime filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic stoploss/take profit on 1h
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
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 1.8x 50-period average (tight to reduce trades)
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
        elif i > 0:
            vol_ma_50 = np.mean(volume[:i])
        else:
            vol_ma_50 = 0
        volume_spike = volume[i] > (1.8 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_ema_4h = ema_21_4h_aligned[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment on both timeframes
            if volume_spike:
                # Bullish entry: price breaks above 4h R1 with 4h uptrend and 1d uptrend regime
                if curr_close > curr_r1 and curr_close > curr_ema_4h and curr_close > curr_ema_1d:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h S1 with 4h downtrend and 1d downtrend regime
                elif curr_close < curr_s1 and curr_close < curr_ema_4h and curr_close < curr_ema_1d:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: price breaks below 4h S1 (reversal signal)
            if curr_close < curr_s1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry
            elif curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: price breaks above 4h R1 (reversal signal)
            if curr_close > curr_r1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry
            elif curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.20
    
    return signals