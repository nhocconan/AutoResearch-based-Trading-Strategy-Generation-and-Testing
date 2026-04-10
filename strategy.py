#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Long when price breaks above Camarilla R4 (4h) AND 4h volume > 1.5x 20-bar avg AND 1d close > 1d open (bullish daily candle)
# - Short when price breaks below Camarilla S4 (4h) AND 4h volume > 1.5x 20-bar avg AND 1d close < 1d open (bearish daily candle)
# - Exit when price returns to Camarilla PP (pivot point) from 4h
# - Session filter: only trade 08-20 UTC to avoid low-volume Asian session
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla levels provide precise support/resistance; volume confirms institutional participation
# - Daily trend filter ensures alignment with higher timeframe momentum, reducing counter-trend whipsaws
# - 1h timeframe used only for entry timing precision, signal direction from 4h/1d

name = "1h_4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 10:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from 4h data (using previous bar's OHLC)
    # Camarilla formulas: PP = (H+L+C)/3, R4 = C + ((H-L)*1.1/2), S4 = C - ((H-L)*1.1/2)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    camarilla_pp = (high_4h + low_4h + close_4h) / 3.0
    camarilla_r4 = close_4h + ((high_4h - low_4h) * 1.1 / 2.0)
    camarilla_s4 = close_4h - ((high_4h - low_4h) * 1.1 / 2.0)
    
    # Align 4h Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume_4h = df_4h['volume'].values
    volume_20_avg = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume_4h > (1.5 * volume_20_avg)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h)
    
    # Pre-compute 1d trend filter: bullish if close > open, bearish if close < open
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_spike_4h_aligned[i]) or
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if outside trading session
        if not session_filter[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Camarilla R4 AND 4h volume spike AND daily bullish
            if (prices['high'].iloc[i] > camarilla_r4_aligned[i] and 
                vol_spike_4h_aligned[i] and 
                daily_bullish_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below Camarilla S4 AND 4h volume spike AND daily bearish
            elif (prices['low'].iloc[i] < camarilla_s4_aligned[i] and 
                  vol_spike_4h_aligned[i] and 
                  daily_bearish_aligned[i]):
                position = -1
                signals[i] = -0.20
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
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals