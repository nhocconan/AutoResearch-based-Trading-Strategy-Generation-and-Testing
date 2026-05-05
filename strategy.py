#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1d volume spike and 1d ADX trend filter
# Long when: price breaks above R4, volume > 2x 20-period average, and 1d ADX > 25
# Short when: price breaks below S4, volume > 2x 20-period average, and 1d ADX > 25
# Exit when price returns to the 1d VWAP (mean reversion) or opposite breakout
# Uses Camarilla levels from 1d for structure and 1d ADX for trend strength, effective in both bull (breakout continuation) and bear (strong trends) markets.
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R4S4_Breakout_1dADX25_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Camarilla levels, ADX, and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX trend filter (min_periods=14 for +DM/-DM, then 14 for ADX)
    if len(high_1d) >= 14:
        up_move = np.diff(high_1d, prepend=high_1d[0])
        down_move = -np.diff(low_1d, append=low_1d[-1])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr1 = np.abs(np.subtract(high_1d, np.roll(low_1d, 1)))
        tr2 = np.abs(np.subtract(np.roll(high_1d, 1), close_1d))
        tr3 = np.abs(np.subtract(np.roll(low_1d, 1), close_1d))
        tr1[0] = 0  # First bar has no previous low
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
        atr_safe = np.where(atr == 0, 1e-10, atr)  # Avoid division by zero
        
        plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_safe
        minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_safe
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
        adx_filter = adx > 25
    else:
        adx_filter = np.zeros(len(close_1d), dtype=bool)
    
    # Calculate 1d VWAP for exit (mean reversion target)
    if len(high_1d) >= 1:
        typical_price = (high_1d + low_1d + close_1d) / 3.0
        pv = typical_price * df_1d['volume'].values
        cum_pv = np.nancumsum(pv)
        cum_volume = np.nancumsum(df_1d['volume'].values)
        vwap = np.divide(cum_pv, cum_volume, out=np.full_like(cum_pv, np.nan), where=cum_volume!=0)
    else:
        vwap = np.full(len(close_1d), np.nan)
    
    # Calculate Camarilla levels from previous 1d bar
    if len(high_1d) >= 2:
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        camarilla_r4 = prev_close + 1.1 * rang * 1.1 / 2
        camarilla_s4 = prev_close - 1.1 * rang * 1.1 / 2
    else:
        camarilla_r4 = np.full(len(close_1d), np.nan)
        camarilla_s4 = np.full(len(close_1d), np.nan)
    
    # Align all 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_filter)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R4, volume filter, and strong trend (ADX > 25)
            if (close[i] > camarilla_r4_aligned[i] and 
                open_price[i] <= camarilla_r4_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                adx_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S4, volume filter, and strong trend (ADX > 25)
            elif (close[i] < camarilla_s4_aligned[i] and 
                  open_price[i] >= camarilla_s4_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  adx_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below VWAP (mean reversion) or breaks below S4 (reversal)
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above VWAP (mean reversion) or breaks above R4 (reversal)
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals