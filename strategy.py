#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# - Donchian(20) from 1d: breakout above upper band = long, below lower band = short
# - 1w EMA(21) trend filter: price > EMA21 for long bias, price < EMA21 for short bias
# - Volume confirmation: current 1d volume > 2.0x 20-period average
# - ATR-based trailing stop: exit long when price < highest_high - 3.0*ATR, exit short when price > lowest_low + 3.0*ATR
# - Designed for 1d timeframe: targets 7-25 trades/year to avoid fee drag
# - Works in bull/bear markets: EMA filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Donchian upper band: highest high over past 20 periods
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low over past 20 periods
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume confirmation
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * avg_volume_20)
    
    # Pre-compute 1d ATR(14) for trailing stop
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0  # for trailing stop
    lowest_low = 0.0    # for trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            if close_1d[i] > highest_high:
                highest_high = close_1d[i]
            # Exit: trailing stop hit OR price re-enters Donchian channel (failed breakout)
            if close_1d[i] < highest_high - 3.0 * atr_14[i] or close_1d[i] < highest_20[i]:
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            if close_1d[i] < lowest_low:
                lowest_low = close_1d[i]
            # Exit: trailing stop hit OR price re-enters Donchian channel (failed breakout)
            if close_1d[i] > lowest_low + 3.0 * atr_14[i] or close_1d[i] > lowest_20[i]:
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Determine trend bias from 1w EMA
                trend_bias_long = close_1d[i] > ema_21_1w_aligned[i]
                trend_bias_short = close_1d[i] < ema_21_1w_aligned[i]
                
                # Breakout long: price closes above upper Donchian band with long bias
                if close_1d[i] > highest_20[i] and trend_bias_long:
                    position = 1
                    entry_price = close_1d[i]
                    highest_high = close_1d[i]
                    signals[i] = 0.25
                # Breakout short: price closes below lower Donchian band with short bias
                elif close_1d[i] < lowest_20[i] and trend_bias_short:
                    position = -1
                    entry_price = close_1d[i]
                    lowest_low = close_1d[i]
                    signals[i] = -0.25
    
    return signals