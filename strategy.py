#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla H3/L3 levels with 1d trend filter and volume confirmation
# Camarilla H3/L3 act as strong weekly support/resistance; breakouts with volume and 1d trend alignment
# capture institutional moves. Designed to work in both bull and bear markets by requiring
# volume confirmation and trend alignment to avoid false breakouts. Target: 50-150 total trades over 4 years.

name = "12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels using prior 1w bar (HLC of previous week)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Camarilla levels based on prior 1w bar (exclude current)
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = np.nan  # first bar has no prior
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla H3 and L3 levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4  # H3
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4  # L3
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1w bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
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
    
    start_idx = 50  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above Camarilla H3 with 1d uptrend
                if curr_close > camarilla_h3_aligned[i] and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Camarilla L3 with 1d downtrend
                elif curr_close < camarilla_l3_aligned[i] and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks Camarilla L3
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla H4
            elif curr_close >= camarilla_h3_aligned[i] + 1.1 * (high_1w[-1] - low_1w[-1]) / 2:  # H4
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks Camarilla H3
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla L4
            elif curr_close <= camarilla_l3_aligned[i] - 1.1 * (high_1w[-1] - low_1w[-1]) / 2:  # L4
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals