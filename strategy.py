#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Choppiness Index + Donchian Breakout + Volume Confirmation
    # Choppiness Index identifies ranging vs trending markets (CHOP > 61.8 = range, < 38.2 = trend)
    # In trending regimes, Donchian breakouts capture momentum with volume confirmation
    # Works in both bull/bear by filtering false breakouts in ranging markets
    # Weekly trend filter ensures alignment with higher timeframe momentum
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index and Weekly trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d Choppiness Index (14-period)
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate True Range for 1d data
    prev_close_1d = np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].values[:-1]])
    tr_1d = true_range(df_1d['high'].values, df_1d['low'].values, prev_close_1d)
    
    # ATR(14) and Sum of True Range
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(SUM(TR14) / (ATR14 * 14)) / log10(14)
    chop_raw = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop_1d = chop_raw  # Already calculated
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 12h Donchian Channels (20-period)
    def donchian_channels(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.3 * vol_ma20  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: Only trade when market is trending (CHOP < 38.2)
        is_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0 and is_trending:
            # Long: Price breaks above Donchian high + volume surge + above weekly EMA50
            if close[i] > donch_hi[i] and vol_surge[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume surge + below weekly EMA50
            elif close[i] < donch_lo[i] and vol_surge[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Trend weakening (CHOP > 50) or Donchian reversal
            if position == 1:
                if chop_1d_aligned[i] > 50 or close[i] < donch_lo[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if chop_1d_aligned[i] > 50 or close[i] > donch_hi[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Choppiness_Donchian_Breakout_1wEMA50_VolumeSurge_v1"
timeframe = "12h"
leverage = 1.0