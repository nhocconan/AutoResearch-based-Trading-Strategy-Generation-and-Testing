#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w pivot high/low with 6h trend filter and volume confirmation
# Weekly pivot levels (based on prior week's high, low, close) identify key institutional support/resistance.
# Breakouts above weekly pivot high or below weekly pivot low with volume spike indicate strong directional moves.
# 6h EMA(50) ensures alignment with intermediate-term trend to avoid counter-trend trades.
# Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.

name = "6h_WeeklyPivot_Breakout_6hTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (based on prior week's high, low, close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (PP)
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    r2 = pp + (high_1w - low_1w)
    s2 = pp - (high_1w - low_1w)
    r3 = high_1w + 2 * (pp - low_1w)
    s3 = low_1w - 2 * (high_1w - pp)
    
    # Align weekly pivot levels to 6h timeframe (wait for completed 1w bar)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 6h EMA(50) for trend filter
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for dynamic stoploss
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
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_ema = ema_50[i]
        curr_atr = atr[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_r2 = r2_aligned[i]
        curr_s2 = s2_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above weekly R1 with 6h uptrend
                if curr_close > curr_r1 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below weekly S1 with 6h downtrend
                elif curr_close < curr_s1 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks weekly S1
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches weekly R2
            elif curr_close >= curr_r2:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks weekly R1
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r1:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches weekly S2
            elif curr_close <= curr_s2:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals