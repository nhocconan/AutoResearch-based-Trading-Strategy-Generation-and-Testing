#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and ADX trend filter
# - Long when price breaks above Camarilla H3 (1d) AND 1d volume > 2x 20-bar avg AND 1d ADX(14) > 25
# - Short when price breaks below Camarilla L3 (1d) AND 1d volume > 2x 20-bar avg AND 1d ADX(14) > 25
# - Exit when price returns to Camarilla pivot point (1d)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla levels provide institutional support/resistance; volume confirms institutional participation
# - ADX filter ensures we only trade strong trends, avoiding choppy markets
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)

name = "4h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h3 = pivot + (range_1d * 1.1 / 4)  # H3 = pivot + 1.1*(HL/4)
    l3 = pivot - (range_1d * 1.1 / 4)  # L3 = pivot - 1.1*(HL/4)
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Pre-compute 1d ADX(14) for trend filter
    # ADX calculation: +DI, -DI, then DX, then smoothed ADX
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[:-1] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])  # TR with NaN for first bar
    
    plus_dm = np.concatenate([[np.nan], np.maximum(high_1d[1:] - high_1d[:-1], 0)])
    minus_dm = np.concatenate([[np.nan], np.maximum(low_1d[:-1] - low_1d[1:], 0)])
    
    # Only update DM when it's greater than the other and positive
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / np.where(atr == 0, 1, atr)
    minus_di = 100 * minus_dm_smooth / np.where(atr == 0, 1, atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    strong_trend = adx_aligned > 25
    
    # Pre-compute 1d volume confirmation: > 2x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * volume_20_avg)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(strong_trend[i]) or np.isnan(vol_spike_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_price = prices['close'].iloc[i]
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND strong trend AND volume spike
            if (close_price > h3_aligned[i] and 
                strong_trend[i] and 
                vol_spike_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND strong trend AND volume spike
            elif (close_price < l3_aligned[i] and 
                  strong_trend[i] and 
                  vol_spike_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion to fair value)
            # Exit when price returns to Camarilla pivot point
            exit_signal = np.abs(close_price - pivot_aligned[i]) < (0.001 * close_price)  # Within 0.1% of pivot
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals