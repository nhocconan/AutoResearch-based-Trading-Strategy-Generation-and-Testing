#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Long when price breaks above Camarilla R4 (12h) AND 12h volume > 1.8x 20-bar avg AND 1d close > 1d open (bullish daily candle)
# - Short when price breaks below Camarilla S4 (12h) AND 12h volume > 1.8x 20-bar avg AND 1d close < 1d open (bearish daily candle)
# - Exit when price returns to Camarilla PP (pivot point) from 12h
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Camarilla levels from 12h provide intermediate-term support/resistance; volume confirms institutional participation
# - Daily trend filter ensures alignment with higher timeframe momentum, reducing counter-trend whipsaws

name = "6h_12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 10:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from 12h data (using previous 12h bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_pp = (high_12h + low_12h + close_12h) / 3.0
    camarilla_r4 = close_12h + ((high_12h - low_12h) * 1.1 / 2.0)
    camarilla_s4 = close_12h - ((high_12h - low_12h) * 1.1 / 2.0)
    
    # Align 12h Camarilla levels to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Pre-compute 12h volume confirmation: > 1.8x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.8 * volume_20_avg)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Pre-compute 1d trend filter: bullish if close > open, bearish if close < open
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Camarilla R4 AND 12h volume spike AND daily bullish
            if (prices['high'].iloc[i] > camarilla_r4_aligned[i] and 
                vol_spike_12h_aligned[i] and 
                daily_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Camarilla S4 AND 12h volume spike AND daily bearish
            elif (prices['low'].iloc[i] < camarilla_s4_aligned[i] and 
                  vol_spike_12h_aligned[i] and 
                  daily_bearish_aligned[i]):
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