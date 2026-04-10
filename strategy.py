#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 (1d) AND 1w close > 1w open (bullish weekly candle) AND 1d volume > 2.0x 20-bar avg
# - Short when price breaks below Camarilla L3 (1d) AND 1w close < 1w open (bearish weekly candle) AND 1d volume > 2.0x 20-bar avg
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Weekly trend filter ensures alignment with higher timeframe momentum to avoid counter-trend trades
# - Volume confirmation threshold increased to 2.0x to reduce false signals and trade frequency
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, mean reversion in ranges

name = "12h_1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * volume_20_avg_1d)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    rang = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_h3 = camarilla_pivot + (rang * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (rang * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (rang * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (rang * 1.1 / 2)
    
    # Pre-compute 1w trend filter: bullish if close > open, bearish if close < open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align HTF indicators to 12h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_1d_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND weekly bullish AND volume spike
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                weekly_bullish_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND weekly bearish AND volume spike
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  weekly_bearish_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion)
            # Exit when price returns to Camarilla pivot point
            exit_long = position == 1 and prices['close'].iloc[i] <= camarilla_pivot_aligned[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= camarilla_pivot_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals