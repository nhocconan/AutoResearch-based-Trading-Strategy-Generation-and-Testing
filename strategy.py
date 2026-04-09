#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining Donchian channel breakout with 12h volume spike and choppiness regime filter
# Donchian(20) breakout provides clear entry/exit signals with proven edge on SOLUSDT
# 12h volume spike (volume > 1.5 * 20-period average) confirms institutional participation
# Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert at Donchian middle), CHOP < 38.2 = trend (follow breakout)
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year and reduce fee drag
# Works in bull/bear markets: trend following in strong trends, mean reversion in ranging markets

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h choppiness index
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[:-1])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over period
        tr_sum = np.zeros(len(close))
        for i in range(period, len(tr)):
            tr_sum[i] = np.nansum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros(len(high))
        ll = np.zeros(len(low))
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop = 100 * log10(sum(tr) / (hh - ll)) / log10(period)
        chop = np.zeros(len(close))
        for i in range(period-1, len(close)):
            if hh[i] > ll[i] and tr_sum[i] > 0:
                chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50.0  # neutral value
        return chop
    
    chop_12h = choppiness_index(high_12h, low_12h, close_12h, 14)
    
    # Calculate 12h volume spike (volume > 1.5 * 20-period average)
    def volume_spike(volume, period=20):
        vol_ma = np.zeros(len(volume))
        for i in range(period-1, len(volume)):
            vol_ma[i] = np.mean(volume[i-period+1:i+1])
        spike = np.zeros(len(volume))
        for i in range(len(volume)):
            if vol_ma[i] > 0:
                spike[i] = volume[i] / vol_ma[i]
            else:
                spike[i] = 1.0
        return spike
    
    vol_spike_12h = volume_spike(volume_12h, 20)
    vol_spike_threshold = 1.5  # volume > 1.5 * average
    
    # Align 12h indicators to 4h timeframe
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Calculate 4h Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.zeros(len(high))
        lower = np.zeros(len(low))
        middle = np.zeros(len(high))
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
            middle[i] = (upper[i] + lower[i]) / 2
        
        # Fill beginning with NaN
        upper[:period-1] = np.nan
        lower[:period-1] = np.nan
        middle[:period-1] = np.nan
        
        return upper, lower, middle
    
    donch_upper, donch_lower, donch_middle = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(donch_middle[i]) or
            np.isnan(chop_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime and volume conditions
        chop_value = chop_12h_aligned[i]
        vol_spike_value = vol_spike_12h_aligned[i]
        
        # Regime classification
        ranging_regime = chop_value > 61.8
        trending_regime = chop_value < 38.2
        neutral_regime = 38.2 <= chop_value <= 61.8
        
        # Volume confirmation
        volume_confirmed = vol_spike_value > vol_spike_threshold
        
        if position == 1:  # Long position
            if ranging_regime:
                # Exit long if price returns to Donchian middle
                if close[i] >= donch_middle[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif trending_regime or neutral_regime:
                # Exit long if price breaks below Donchian lower
                if close[i] <= donch_lower[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if ranging_regime:
                # Exit short if price returns to Donchian middle
                if close[i] <= donch_middle[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif trending_regime or neutral_regime:
                # Exit short if price breaks above Donchian upper
                if close[i] >= donch_upper[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                if ranging_regime:
                    # Mean revert at Donchian extremes in ranging market
                    if close[i] <= donch_lower[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] >= donch_upper[i]:
                        position = -1
                        signals[i] = -0.25
                elif trending_regime:
                    # Follow breakout in trending market
                    if close[i] >= donch_upper[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] <= donch_lower[i]:
                        position = -1
                        signals[i] = -0.25
                # In neutral regime, wait for clearer signal
    
    return signals