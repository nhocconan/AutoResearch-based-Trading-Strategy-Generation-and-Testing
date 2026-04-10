#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + chop regime filter
# - Long when price breaks above Donchian(20) upper band AND volume > 1.3x 20-bar avg AND chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) lower band AND volume > 1.3x 20-bar avg AND chop < 61.8 (trending)
# - Exit when price crosses opposite Donchian band or chop > 61.8 (range) to avoid false breakouts
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian breakouts capture momentum; volume confirmation avoids low-liquidity fakes; chop filter ensures trending conditions
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue); chop filter avoids whipsaws in ranges

name = "4h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Pre-compute Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (highest_high - lowest_low)) / log10(n)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop = np.where(
        (chop_denom > 0) & ~np.isnan(atr_sum),
        100 * np.log10(atr_sum / (chop_denom * np.log10(14))) / np.log10(14),
        50  # default to neutral when invalid
    )
    chop_filter = chop < 61.8  # trending regime
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.3 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(chop_filter[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND volume spike AND trending
            if (close[i] > donchian_upper[i-1] and  # break above prior bar's upper band
                vol_spike[i] and 
                chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND volume spike AND trending
            elif (close[i] < donchian_lower[i-1] and  # break below prior bar's lower band
                  vol_spike[i] and 
                  chop_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when price crosses opposite Donchian band OR chop > 61.8 (range)
            exit_signal = False
            if position == 1:  # long position
                if close[i] < donchian_lower[i]:  # cross below lower band
                    exit_signal = True
                elif not chop_filter[i]:  # chop > 61.8 (range)
                    exit_signal = True
            else:  # short position
                if close[i] > donchian_upper[i]:  # cross above upper band
                    exit_signal = True
                elif not chop_filter[i]:  # chop > 61.8 (range)
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