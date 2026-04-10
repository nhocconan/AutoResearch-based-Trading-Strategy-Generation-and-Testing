#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h ATR-based volatility filter + chop regime
# - Primary: 4h Donchian channel breakout (20-period) for directional bias
# - HTF: 12h ATR ratio (current ATR(7) / ATR(30)) > 1.2 for volatility expansion + chop regime (CHOP < 50 = trending)
# - Long: Price breaks above Donchian upper band + volatility expansion + chop regime (trending)
# - Short: Price breaks below Donchian lower band + volatility expansion + chop regime (trending)
# - Exit: Opposite Donchian breakout or chop regime shifts to ranging (CHOP > 60)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures institutional breakouts, volatility filter confirms momentum, chop filter avoids false signals in ranging markets
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe

name = "4h_12h_donchian_volatility_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(7) for volatility filter
    atr_period_short = 7
    tr_12h = np.maximum(np.maximum(high_12h - low_12h,
                                  np.abs(np.roll(high_12h, 1) - low_12h)),
                       np.abs(np.roll(low_12h, 1) - high_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]  # First TR
    atr_7_12h = np.full(len(tr_12h), np.nan)
    for i in range(atr_period_short, len(tr_12h)):
        if not np.isnan(tr_12h[i-atr_period_short+1:i+1]).any():
            atr_7_12h[i] = np.mean(tr_12h[i-atr_period_short+1:i+1])
    
    # Calculate 12h ATR(30) for volatility filter
    atr_period_long = 30
    atr_30_12h = np.full(len(tr_12h), np.nan)
    for i in range(atr_period_long, len(tr_12h)):
        if not np.isnan(tr_12h[i-atr_period_long+1:i+1]).any():
            atr_30_12h[i] = np.mean(tr_12h[i-atr_period_long+1:i+1])
    
    # ATR ratio (current ATR(7) / ATR(30)) for volatility expansion
    atr_ratio_12h = np.full(len(tr_12h), np.nan)
    for i in range(len(tr_12h)):
        if not np.isnan(atr_7_12h[i]) and not np.isnan(atr_30_12h[i]) and atr_30_12h[i] > 0:
            atr_ratio_12h[i] = atr_7_12h[i] / atr_30_12h[i]
    
    # Calculate 12h Chopiness Index (CHOP) for regime filter
    chop_lookback = 14
    sum_tr = np.full(len(tr_12h), np.nan)
    for i in range(chop_lookback, len(tr_12h)):
        if not np.isnan(tr_12h[i-chop_lookback:i]).any():
            sum_tr[i] = np.sum(tr_12h[i-chop_lookback:i])
    
    hh = np.full(len(high_12h), np.nan)
    ll = np.full(len(low_12h), np.nan)
    for i in range(chop_lookback, len(high_12h)):
        if (not np.isnan(high_12h[i-chop_lookback:i+1]).any() and 
            not np.isnan(low_12h[i-chop_lookback:i+1]).any()):
            hh[i] = np.max(high_12h[i-chop_lookback:i+1])
            ll[i] = np.min(low_12h[i-chop_lookback:i+1])
    
    chop = np.full(len(high_12h), np.nan)
    for i in range(chop_lookback, len(high_12h)):
        if (not np.isnan(sum_tr[i]) and not np.isnan(hh[i]) and not np.isnan(ll[i]) and 
            hh[i] > ll[i] and sum_tr[i] > 0):
            chop[i] = 100 * np.log10(sum_tr[i] / (hh[i] - ll[i])) / np.log10(chop_lookback)
        else:
            chop[i] = np.nan
    
    # Align all HTF indicators to 4h timeframe
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_period = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(donchian_period-1, n):
        upper_band[i] = np.max(high_4h[i-donchian_period+1:i+1])
        lower_band[i] = np.min(low_4h[i-donchian_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from second bar to avoid index issues
        # Skip if any required data is invalid
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(atr_ratio_12h_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion: ATR ratio > 1.2
        volatility_confirm = atr_ratio_12h_aligned[i] > 1.2
        
        # Chop regime filter: CHOP < 50 indicates trending market (avoid ranging)
        regime_confirm = chop_aligned[i] < 50.0
        
        # Donchian breakout signals
        donchian_up = close_4h[i] > upper_band[i]
        donchian_down = close_4h[i] < lower_band[i]
        
        # Exit conditions: Opposite Donchian breakout or chop regime shifts to ranging (CHOP > 60)
        exit_long = donchian_down or (chop_aligned[i] > 60.0)
        exit_short = donchian_up or (chop_aligned[i] > 60.0)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Donchian breakout up + volatility expansion + trending regime
            if donchian_up and volatility_confirm and regime_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian breakout down + volatility expansion + trending regime
            elif donchian_down and volatility_confirm and regime_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite Donchian breakout OR chop regime shifts to ranging
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals