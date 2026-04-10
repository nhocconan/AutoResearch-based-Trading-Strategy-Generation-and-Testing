#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with volume confirmation and 1w trend filter
# - Long when price breaks above Camarilla R4 (1d) AND daily volume > 1.5x 20-day avg AND weekly close > weekly open
# - Short when price breaks below Camarilla S4 (1d) AND daily volume > 1.5x 20-day avg AND weekly close < weekly open
# - Exit when price returns to Camarilla PP (pivot point) from 1d
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla levels provide precise support/resistance; volume confirms institutional participation
# - Weekly trend filter ensures alignment with higher timeframe momentum, reducing counter-trend whipsaws

name = "1d_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from 1d data (using previous day's OHLC)
    # Camarilla formulas: PP = (H+L+C)/3, R4 = C + ((H-L)*1.1/2), S4 = C - ((H-L)*1.1/2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2.0)
    camarilla_s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 1d timeframe (no shift needed as already aligned)
    camarilla_pp_aligned = camarilla_pp  # Already on 1d timeframe
    camarilla_r4_aligned = camarilla_r4
    camarilla_s4_aligned = camarilla_s4
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg)
    
    # Pre-compute 1w trend filter: bullish if close > open, bearish if close < open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_spike_1d[i]) or
            np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Camarilla R4 AND volume spike AND weekly bullish
            if (prices['high'].iloc[i] > camarilla_r4_aligned[i] and 
                vol_spike_1d[i] and 
                weekly_bullish[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Camarilla S4 AND volume spike AND weekly bearish
            elif (prices['low'].iloc[i] < camarilla_s4_aligned[i] and 
                  vol_spike_1d[i] and 
                  weekly_bearish[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Camarilla PP (mean reversion to equilibrium)
            # Exit when price returns to Camarilla pivot point
            exit_signal = False
            if position == 1:  # Long position
                if prices['low'].iloc[i] <= camarilla_pp_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['high'].iloc[i] >= camarilla_pp_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals