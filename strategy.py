#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with 1d trend filter and volume confirmation
    # Camarilla pivot levels identify key support/resistance from prior 1d range
    # Breakout above H3 or below L3 with volume confirmation and 1d trend alignment
    # Designed for low trade frequency (target: 12-37/year) to minimize fee drag
    # Works in bull/bear markets by only trading breakouts in direction of higher trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate prior 12h Camarilla levels (H3, L3, H4, L4)
    # Based on prior 12h candle's high-low range
    camarilla_h3 = np.full_like(close_12h, np.nan)
    camarilla_l3 = np.full_like(close_12h, np.nan)
    camarilla_h4 = np.full_like(close_12h, np.nan)
    camarilla_l4 = np.full_like(close_12h, np.nan)
    camarilla_close = np.full_like(close_12h, np.nan)
    
    for i in range(1, len(df_12h)):
        # Prior 12h candle
        phigh = high_12h[i-1]
        plow = low_12h[i-1]
        pclose = close_12h[i-1]
        range_val = phigh - plow
        
        if range_val > 0:
            camarilla_h3[i] = pclose + range_val * 1.1 / 4
            camarilla_l3[i] = pclose - range_val * 1.1 / 4
            camarilla_h4[i] = pclose + range_val * 1.1 / 2
            camarilla_l4[i] = pclose - range_val * 1.1 / 2
            camarilla_close[i] = pclose
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h volume for confirmation
    vol_ma_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Volume confirmation: volume > 1.8 * 20-period average (12h)
    volume_spike_12h = volume_12h > (1.8 * vol_ma_12h)
    
    # Align all indicators to LTF
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # 1d trend filter
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Camarilla breakout + trend filter + volume confirmation
        long_entry = False
        short_entry = False
        
        # Long: break above H3 (or H4) with bullish trend and volume spike
        if bullish_trend and volume_spike_aligned[i]:
            if close[i] > h3_aligned[i] or close[i] > h4_aligned[i]:
                long_entry = True
        # Short: break below L3 (or L4) with bearish trend and volume spike
        elif bearish_trend and volume_spike_aligned[i]:
            if close[i] < l3_aligned[i] or close[i] < l4_aligned[i]:
                short_entry = True
        
        # Exit logic: trend reversal or price returns to Camarilla equilibrium (close)
        long_exit = bearish_trend or (close[i] < close_12h_aligned[i] and position == 1)
        short_exit = bullish_trend or (close[i] > close_12h_aligned[i] and position == -1)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_h3l3_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0