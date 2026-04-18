# 6h_WeeklyPivot_Direction_1dTrend_4hVolume
# Hypothesis: Weekly pivot direction determines bias (above/below pivot = long/short bias).
# Daily EMA(50) filters for trend alignment. 4h volume spike confirms momentum.
# Weekly pivots are strong institutional levels; breakouts with volume and trend alignment
# capture momentum in both bull and bear markets. Target: 15-25 trades/year on 6h.

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
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using standard formula
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    pivot_w = (high_1w + low_1w + close_1w) / 3
    r1_w = 2 * pivot_w - low_1w
    s1_w = 2 * pivot_w - high_1w
    
    # Shift by 1 to use previous week's levels only
    pivot_w_prev = pivot_w.shift(1).values
    r1_w_prev = r1_w.shift(1).values
    s1_w_prev = s1_w.shift(1).values
    
    # Align to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w_prev)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w_prev)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w_prev)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(close_1d := df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for volatility filter (14-period on 6h)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 20-period average (avoid chop)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_ma
    
    # Volume spike: 2.5x 20-period average on 4h (use 4h volume for confirmation)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (2.5 * vol_ma_4h)
    # Align 4h volume spike to 6h timeframe
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    bars_since_entry = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or
            np.isnan(s1_w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volatility_filter[i]) or
            np.isnan(volume_spike_4h_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        pivot_val = pivot_w_aligned[i]
        r1_val = r1_w_aligned[i]
        s1_val = s1_w_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike_4h_aligned[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: price above weekly pivot AND above weekly R1 with volume spike and above daily EMA
            if price > pivot_val and price > r1_val and vol_spike and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND below weekly S1 with volume spike and below daily EMA
            elif price < pivot_val and price < s1_val and vol_spike and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Minimum holding period: 3 bars (18 hours for 6h)
            if bars_since_entry < 3:
                signals[i] = 0.25
                bars_since_entry += 1
            else:
                signals[i] = 0.25
                # Exit: price returns to weekly pivot or breaks below daily EMA
                if price <= pivot_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            # Minimum holding period: 3 bars (18 hours for 6h)
            if bars_since_entry < 3:
                signals[i] = -0.25
                bars_since_entry += 1
            else:
                signals[i] = -0.25
                # Exit: price returns to weekly pivot or breaks above daily EMA
                if price >= pivot_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "6h_WeeklyPivot_Direction_1dTrend_4hVolume"
timeframe = "6h"
leverage = 1.0