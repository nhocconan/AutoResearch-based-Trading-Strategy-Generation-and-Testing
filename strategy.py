#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for structural bias (long when price > EMA34, short when price < EMA34)
# Donchian breakout provides entry timing in direction of 1d trend
# Volume confirmation > 1.5x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~20-50 trades/year per symbol with 0.25 sizing
# 1d EMA34 acts as dynamic trend filter that works in both bull and bear markets
# Breakouts in direction of 1d EMA34 have higher follow-through probability

name = "4h_Donchian20_1dEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian(20) breakout
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().shift(1).values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d EMA34 (34 days) + Donchian20 + volume EMA20
    start_idx = max(34*6, donchian_window, 20)  # 34 days * 6 (4h bars per day) = 204
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine bias from 1d EMA34: long above EMA, short below EMA
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Donchian breakout above upper band with volume spike
                if close[i] > donchian_high[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Donchian breakdown below lower band with volume spike
                if close[i] < donchian_low[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown below lower band (failure of breakout)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout above upper band (failure of breakdown)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals