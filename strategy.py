#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
    # Works in bull/bear by trading only when price respects Camarilla levels + volume confirms breakout
    # Choppiness filter avoids whipsaws in ranging markets. Target: 20-40 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Align 1d ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Previous 1d Camarilla levels (H3/L3 for breakout, H4/L4 for extreme)
    # Using previous day's range to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    range_1d = prev_high_1d - prev_low_1d
    camarilla_h3 = prev_close_1d + range_1d * 1.1 / 4
    camarilla_l3 = prev_close_1d - range_1d * 1.1 / 4
    camarilla_h4 = prev_close_1d + range_1d * 1.1 / 2
    camarilla_l4 = prev_close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(33, len(df_1d)):
        if not np.isnan(np.mean(volume_1d[i-19:i+1])):
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = (not np.isnan(vol_ma_20_1d_aligned) & 
                   (volume_1d > 2.0 * vol_ma_20_1d))
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # 1d Choppiness Index: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    # Using 14-period CHOP
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_sum = np.sum(tr[i-14:i+1])
        max_high = np.max(high_1d[i-14:i+1])
        min_low = np.min(low_1d[i-14:i+1])
        if max_high > min_low and atr_sum > 0:
            chop_1d[i] = 100 * np.log10(atr_sum / np.log10(max_high - min_low)) / np.log10(14)
        else:
            chop_1d[i] = np.nan
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_filter = chop_1d_aligned < 61.8  # Avoid strong ranging markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions: price breaks Camarilla H3/L3 with volume spike
        breakout_long = close[i] > h3_4h[i] and volume_spike_aligned[i] > 0.5
        breakout_short = close[i] < l3_4h[i] and volume_spike_aligned[i] > 0.5
        
        # Additional filter: avoid trading in choppy markets
        trend_filter = chop_1d_aligned[i] < 61.8
        
        # Entry conditions
        long_entry = breakout_long and trend_filter
        short_entry = breakout_short and trend_filter
        
        # Exit conditions: opposite breakout or loss of volume/spike
        long_exit = (close[i] < l3_4h[i]) or (volume_spike_aligned[i] <= 0.5)
        short_exit = (close[i] > h3_4h[i]) or (volume_spike_aligned[i] <= 0.5)
        
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

name = "4h_1d_camarilla_breakout_volume_chop_filter_v1"
timeframe = "4h"
leverage = 1.0