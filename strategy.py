#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla H3/L3 levels with 12h trend filter and volume confirmation
# Weekly Camarilla levels identify key weekly support/resistance where institutional order flow clusters.
# Breakouts above H3 or below L3 with volume spike indicate strong institutional participation.
# 12h EMA(34) ensures alignment with medium-term trend to avoid counter-trend trades.
# Designed for low trade frequency (<25/year) to minimize fee drag in both bull and bear markets.
# Uses 6h timeframe as requested, with 1w HTF for Camarilla levels and 12h HTF for trend filter.

name = "6h_WeeklyCamarilla_H3L3_Breakout_12hTrend_VolumeSpike_v1"
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
    
    # Load weekly data ONCE before loop for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (H3, L3, H4, L4)
    # Based on previous week's high, low, close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Calculate Camarilla levels
    h3 = pp + (high_1w - low_1w) * 1.1 / 4.0
    l3 = pp - (high_1w - low_1w) * 1.1 / 4.0
    h4 = pp + (high_1w - low_1w) * 1.1 / 2.0
    l4 = pp - (high_1w - low_1w) * 1.1 / 2.0
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed weekly bar)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h_s = pd.Series(df_12h['close'].values)
    ema_34_12h = close_12h_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
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
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_ema = ema_34_12h_aligned[i]
        curr_atr = atr[i]
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_h4 = h4_aligned[i]
        curr_l4 = l4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above weekly Camarilla H3 with 12h uptrend
                if curr_close > curr_h3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below weekly Camarilla L3 with 12h downtrend
                elif curr_close < curr_l3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks weekly Camarilla L3
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_l3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches weekly Camarilla H4
            elif curr_close >= curr_h4:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks weekly Camarilla H3
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_h3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches weekly Camarilla L4
            elif curr_close <= curr_l4:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals