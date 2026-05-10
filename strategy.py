#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Reversal
# Hypothesis: Fade Camarilla R1/S1 reversals with 1d trend filter and volume confirmation.
# Uses mean-reversion at pivot levels in ranging markets, filtered by daily trend.
# Designed for 20-40 trades/year to minimize fee drag. Works in sideways/low-vol regimes.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for dynamic sizing and stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    camarilla_R2 = np.full(n, np.nan)
    camarilla_S2 = np.full(n, np.nan)
    
    # Need previous day's OHLC
    prev_day_high = np.roll(high, 1)
    prev_day_low = np.roll(low, 1)
    prev_day_close = np.roll(close, 1)
    prev_day_high[0] = np.nan
    prev_day_low[0] = np.nan
    prev_day_close[0] = np.nan
    
    for i in range(1, n):
        if np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(prev_day_close[i]):
            continue
        rng = prev_day_high[i] - prev_day_low[i]
        camarilla_R1[i] = prev_day_close[i] + rng * 1.1 / 12
        camarilla_S1[i] = prev_day_close[i] - rng * 1.1 / 12
        camarilla_R2[i] = prev_day_close[i] + rng * 1.1 / 6
        camarilla_S2[i] = prev_day_close[i] - rng * 1.1 / 6
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when price is between R2 and S2 (range-bound)
        in_range = camarilla_S2[i] < close[i] < camarilla_R2[i]
        
        if position == 0 and in_range:
            # Fade R1/S1 with volume confirmation
            if close[i] <= camarilla_S1[i] and volume[i] > 1.5 * vol_ma[i]:
                # Long at S1 bounce
                signals[i] = 0.25
                position = 1
            elif close[i] >= camarilla_R1[i] and volume[i] > 1.5 * vol_ma[i]:
                # Short at R1 rejection
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches R2 or trend changes
            if close[i] >= camarilla_R2[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches S2 or trend changes
            if close[i] <= camarilla_S2[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals