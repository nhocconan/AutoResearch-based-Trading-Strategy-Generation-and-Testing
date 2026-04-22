#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian channel breakout with 12h volume confirmation and 1d EMA50 trend filter
    # Donchian channels identify breakouts from volatility contractions. Volume confirms institutional participation.
    # 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
    # This combination reduces false breakouts and improves win rate in both bull and bear markets.
    # Focus on 6h timeframe with strict entry conditions to limit trades to 12-37/year.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Donchian channels (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian Channels (20-period)
    dc_period = 20
    upper_dc = pd.Series(high_6h).rolling(window=dc_period, min_periods=dc_period).max().values
    lower_dc = pd.Series(low_6h).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Align Donchian Channels to 6h timeframe (no additional delay needed as Donchian uses current bar)
    upper_dc_aligned = align_htf_to_ltf(prices, df_6h, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_6h, lower_dc)
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    vol_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma20_12h)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready or outside session
        if (np.isnan(upper_dc_aligned[i]) or np.isnan(lower_dc_aligned[i]) or
            np.isnan(vol_ma20_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian with volume + price above 1d EMA50 (uptrend)
            if high[i] > upper_dc_aligned[i] and volume[i] > 1.5 * vol_ma20_12h_aligned[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Donchian with volume + price below 1d EMA50 (downtrend)
            elif low[i] < lower_dc_aligned[i] and volume[i] > 1.5 * vol_ma20_12h_aligned[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian band or trend reversal vs 1d EMA50
            if position == 1:
                if low[i] < lower_dc_aligned[i] or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if high[i] > upper_dc_aligned[i] or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_12hVolume_1dEMA50_Session_v1"
timeframe = "6h"
leverage = 1.0