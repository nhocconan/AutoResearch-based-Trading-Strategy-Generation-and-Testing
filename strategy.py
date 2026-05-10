# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Price breaks Camarilla R1 (long) or S1 (short) levels calculated from prior 12h session,
# with confirmation from 1d EMA50 trend and volume spike. Camarilla levels provide high-probability
# reversal/breakout points in ranging markets, while EMA50 trend filter ensures alignment with higher
# timeframe direction. Volume confirmation reduces false breakouts. Designed to work in both bull and
# bear markets by requiring trend alignment and volume confirmation, reducing false signals during
# choppy periods.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla levels (using prior completed 12h bar)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar: R1, S1
    # R1 = close + (high - low) * 1.12
    # S1 = close - (high - low) * 1.12
    camarilla_r1_12h = close_12h + (high_12h - low_12h) * 0.12  # Corrected multiplier from 1.12 to 0.12
    camarilla_s1_12h = close_12h - (high_12h - low_12h) * 0.12  # Corrected multiplier from 1.12 to 0.12
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar to close)
    camarilla_r1_4h = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma_20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma_20_1d[i] = (vol_sma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or \
           np.isnan(camarilla_r1_4h[i]) or np.isnan(camarilla_s1_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        # Approximate 4h volume from 1d: 1d volume / 6 (since 24h/4h = 6)
        vol_4h_approx = vol_sma_20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Break above Camarilla R1 with uptrend and volume
            if close[i] > camarilla_r1_4h[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 with downtrend and volume
            elif close[i] < camarilla_s1_4h[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA50 (trend reversal)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA50 (trend reversal)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals