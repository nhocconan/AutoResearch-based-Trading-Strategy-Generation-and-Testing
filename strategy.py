#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Weekly Camarilla R3/S3 breakout with volume confirmation and 1w EMA(50) trend filter
# Weekly Camarilla levels provide structural support/resistance for major trend shifts.
# Volume confirmation on 12h ensures participation, 1w EMA(50) aligns with longer-term trend.
# Designed for low trade frequency (~12-25/year) to minimize fee drag and improve bear market performance.
# Uses 12h timeframe with 1w HTF for structure and trend filter.

name = "12h_WeeklyCamarilla_R3S3_Breakout_1wEMA50_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels (R3, S3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True range for Camarilla calculation
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]  # first bar
    atr_1w = pd.Series(tr_1w).ewm(span=5, adjust=False, min_periods=5).mean().values  # ATR(5) for Camarilla
    
    # Calculate Camarilla levels using previous week's data
    camarilla_r3 = np.zeros_like(close_1w)
    camarilla_s3 = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        camarilla_r3[i] = close_1w[i-1] + 1.1 * (high_1w[i-1] - low_1w[i-1]) / 6
        camarilla_s3[i] = close_1w[i-1] - 1.1 * (high_1w[i-1] - low_1w[i-1]) / 6
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for dynamic stoploss on 12h
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
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_ema = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1w Camarilla R3 with 1w uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1w Camarilla S3 with 1w downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1w Camarilla S3
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1w Camarilla R4 level
            camarilla_r4 = close_1w_s.ewm(span=5, adjust=False, min_periods=5).mean().values[-1] + 1.1 * (high_1w_s.ewm(span=5, adjust=False, min_periods=5).mean().values[-1] - low_1w_s.ewm(span=5, adjust=False, min_periods=5).mean().values[-1]) * 1.1 / 2 if len(close_1w) > 0 else curr_close
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, np.full(len(close_1w), camarilla_r4))
            if curr_close >= camarilla_r4_aligned[i]:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1w Camarilla R3
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1w Camarilla S4 level
            camarilla_s4 = close_1w_s.ewm(span=5, adjust=False, min_periods=5).mean().values[-1] - 1.1 * (high_1w_s.ewm(span=5, adjust=False, min_periods=5).mean().values[-1] - low_1w_s.ewm(span=5, adjust=False, min_periods=5).mean().values[-1]) * 1.1 / 2 if len(close_1w) > 0 else curr_close
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, np.full(len(close_1w), camarilla_s4))
            if curr_close <= camarilla_s4_aligned[i]:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals