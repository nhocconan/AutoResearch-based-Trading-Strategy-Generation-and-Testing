#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout + Volume Confirmation + ADX Trend Filter
# Hypothesis: Breakouts of Donchian channels on 12h timeframe with volume confirmation and
# trend strength from daily ADX capture sustained moves in both bull and bear markets.
# Uses tight entry conditions to limit trades and avoid fee drag.

name = "12h_donchian_breakout_volume_adx_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian Channel (20-period) on daily
    donch_period = 20
    upper_donch = pd.Series(high_1d).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low_1d).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # ADX (14-period) on daily for trend strength
    adx_period = 14
    tr1 = pd.Series(high_1d).rolling(2).max() - pd.Series(low_1d).rolling(2).min()
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=adx_period, min_periods=adx_period).mean()
    
    plus_dm = pd.Series(high_1d).diff()
    minus_dm = pd.Series(low_1d).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    plus_di = 100 * (plus_dm.rolling(window=adx_period, min_periods=adx_period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=adx_period, min_periods=adx_period).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Volume average (20-period) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    upper_donch_aligned = align_htf_to_ltf(prices, df_1d, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_1d, lower_donch)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_donch_aligned[i]) or np.isnan(lower_donch_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian or trend weakens
            if close[i] < lower_donch_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian or trend weakens
            if close[i] > upper_donch_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_confirm and trend_filter:
                # Breakout above upper Donchian
                if close[i] > upper_donch_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower Donchian
                elif close[i] < lower_donch_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals