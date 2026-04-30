#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels with volume spike confirmation
# Camarilla R3/S3 levels act as mean reversion zones in ranging markets, while R4/S4 breaks
# indicate strong momentum continuation. Weekly pivots provide structure, volume confirms
# participation. Designed for low trade frequency (~10-25/year) to minimize fee drag.
# Uses 6h timeframe with 1w HTF for Camarilla calculation and 1d EMA(34) for trend filter.

name = "6h_WeeklyCamarilla_R3S3_R4S4_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # Camarilla formula: 
    # H4 = close + 1.1*(high-low)/2
    # L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6
    # L2 = close - 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12
    # L1 = close - 1.1*(high-low)/12
    # We'll use R3=H3, S3=L3, R4=H4, S4=L4
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    rng = high_1w - low_1w
    camarilla_h4 = close_1w + 1.1 * rng / 2.0  # R4
    camarilla_l4 = close_1w - 1.1 * rng / 2.0  # S4
    camarilla_h3 = close_1w + 1.1 * rng / 4.0  # R3
    camarilla_l3 = close_1w - 1.1 * rng / 4.0  # S3
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1w bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Load 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
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
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_h4 = camarilla_h4_aligned[i]
        curr_l4 = camarilla_l4_aligned[i]
        curr_h3 = camarilla_h3_aligned[i]
        curr_l3 = camarilla_l3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above weekly R4 with 1d uptrend
                if curr_close > curr_h4 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below weekly S4 with 1d downtrend
                elif curr_close < curr_l4 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                # Mean reversion longs at S3 in downtrend
                elif curr_close <= curr_l3 and curr_close < curr_ema:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Mean reversion shorts at R3 in uptrend
                elif curr_close >= curr_h3 and curr_close > curr_ema:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks weekly S4
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_l4:
                signals[i] = 0.0
                position = 0
            # Take profit at weekly H3 for mean reversion, or H4 for breakout
            elif curr_close >= curr_h3 and curr_close < curr_ema:  # mean reversion TP
                signals[i] = 0.0
                position = 0
            elif curr_close >= curr_h4:  # breakout TP
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks weekly H4
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_h4:
                signals[i] = 0.0
                position = 0
            # Take profit at weekly L3 for mean reversion, or L4 for breakout
            elif curr_close <= curr_l3 and curr_close > curr_ema:  # mean reversion TP
                signals[i] = 0.0
                position = 0
            elif curr_close <= curr_l4:  # breakout TP
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals