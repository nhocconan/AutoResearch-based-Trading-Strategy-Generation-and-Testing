#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume spike + chop regime filter
# Targets 50-150 total trades over 4 years (12-37/year) on 12h timeframe to minimize fee drag
# Donchian(20) provides clear price channel breakout structure
# 1d EMA50 determines trend bias: long when price > EMA50, short when price < EMA50
# Volume spike (2x 30-period average) confirms institutional participation
# Chop regime filter (CHOP > 61.8) avoids whipsaws in ranging markets
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk
# Uses 12h primary timeframe as specified in experiment #117305

name = "12h_Donchian20_1dEMA50_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period lookback)
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Calculate 1d EMA50 trend (prior completed 1d bar's EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h volume spike (2x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 12h Choppiness Index (CHOP) for regime filter
    # CHOP > 61.8 = ranging market (avoid breakout trades)
    # CHOP < 38.2 = trending market (favor breakout trades)
    chop_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).mean().values
    
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    sum_tr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    range_hl = max_high - min_low
    chop = np.where(
        (range_hl != 0) & (sum_tr != 0),
        100 * np.log10(sum_tr / range_hl) / np.log10(chop_period),
        50.0  # neutral when undefined
    )
    chop_filter = chop > 61.8  # True when ranging (avoid trades)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(donchian_period, 50, 30, chop_period)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Only trade in trending markets (CHOP <= 61.8)
            if not chop_filter[i]:
                # Long entry: price breaks above upper Donchian channel AND price > 1d EMA50 (bullish bias) AND volume spike
                if (close[i] > upper_channel[i] and 
                    close[i] > ema_50_aligned[i] and 
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Short entry: price breaks below lower Donchian channel AND price < 1d EMA50 (bearish bias) AND volume spike
                elif (close[i] < lower_channel[i] and 
                      close[i] < ema_50_aligned[i] and 
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid breakout trades in ranging markets
        
        elif position == 1:  # Long position
            # Exit: price falls below lower Donchian channel OR below 1d EMA50 (trend change)
            if close[i] < lower_channel[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above upper Donchian channel OR above 1d EMA50 (trend change)
            if close[i] > upper_channel[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals