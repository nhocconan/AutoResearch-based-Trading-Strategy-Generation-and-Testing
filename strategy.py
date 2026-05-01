#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX(14) trend strength + 12h Donchian(20) breakout direction + volume confirmation.
# Long when: ADX>25 (trending) AND price breaks above 12h Donchian upper band AND volume > 1.5x 20-bar 6h MA.
# Short when: ADX>25 AND price breaks below 12h Donchian lower band AND volume confirmation.
# Exit when: ADX<20 (range) OR price returns to 12h Donchian midpoint.
# Uses discrete sizing 0.25 to minimize fee churn. Works in bull/bear by capturing strong trends.

name = "6h_ADX_Trend_DonchianBreakout_12h_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h Donchian(20) channels
    highest_high_12h = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_12h = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_12h_high = align_htf_to_ltf(prices, df_12h, highest_high_12h)
    donchian_12h_low = align_htf_to_ltf(prices, df_12h, lowest_low_12h)
    donchian_12h_mid = (donchian_12h_high + donchian_12h_low) / 2.0
    
    # 6h ADX(14) for trend strength
    # Calculate +DM, -DM, TR
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(low, prepend=low[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Volume confirmation: 6h volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period*2)  # warmup for ADX and Donchian
    
    for i in range(start_idx, n):
        # Session filter: 00-24 UTC (6h bars less session-dependent, but avoid extreme lows)
        hour = hours[i]
        if hour < 0 or hour > 23:  # always true, kept for structure
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if np.isnan(adx[i]) or np.isnan(donchian_12h_high[i]) or np.isnan(donchian_12h_low[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_adx = adx[i]
        curr_vol_confirm = volume_confirm[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: ADX>25 (strong trend) AND break above 12h Donchian high AND volume confirmation
            if (curr_adx > 25.0 and 
                curr_high > donchian_12h_high[i] and 
                curr_vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: ADX>25 AND break below 12h Donchian low AND volume confirmation
            elif (curr_adx > 25.0 and 
                  curr_low < donchian_12h_low[i] and 
                  curr_vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: ADX<20 (losing trend) OR price returns to 12h Donchian midpoint
            if (curr_adx < 20.0 or 
                curr_close >= donchian_12h_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: ADX<20 OR price returns to 12h Donchian midpoint
            if (curr_adx < 20.0 or 
                curr_close <= donchian_12h_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals