#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike (>2x 20-bar MA)
# Camarilla pivot levels (R1, R2, S1, S2) from 1d provide institutional support/resistance
# Long when price breaks above R1 with volume spike and price above 1d EMA34
# Short when price breaks below S1 with volume spike and price below 1d EMA34
# Uses 1d EMA34 for higher-timeframe trend alignment to reduce whipsaws in ranging markets.
# Volume spike confirms institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits.
# Works in both bull (breakouts continue trend) and bear (mean reversion at extreme levels).

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for Camarilla pivot and EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d OHLC)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid look-ahead: use only completed 1d bar data
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    r1 = camarilla_pivot + camarilla_range * 1.1 / 12.0
    s1 = camarilla_pivot - camarilla_range * 1.1 / 12.0
    r2 = camarilla_pivot + camarilla_range * 1.1 / 6.0
    s2 = camarilla_pivot - camarilla_range * 1.1 / 6.0
    
    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1d EMA(34) on 1d close for trend filter
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_1d_34_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R1 with volume spike and price above 1d EMA34
            if curr_high > r1_aligned[i] and curr_close > r1_aligned[i] and vol_spike and curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and price below 1d EMA34
            elif curr_low < s1_aligned[i] and curr_close < s1_aligned[i] and vol_spike and curr_close < ema_1d_34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below S1 (mean reversion) or below 1d EMA34 (trend change)
            if curr_close < s1_aligned[i] or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above R1 (mean reversion) or above 1d EMA34 (trend change)
            if curr_close > r1_aligned[i] or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals