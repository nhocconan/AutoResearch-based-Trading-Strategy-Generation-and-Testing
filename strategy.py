#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and chop regime filter
# - Primary: 4h Donchian channel breakout (20-period) for trend following
# - HTF: 1d volume spike (current volume > 2.0x 20-period MA) for conviction
# - Regime: 4h Choppiness Index > 61.8 for ranging markets (mean reversion at Donchian mid)
# - Long: Price > Donchian High + volume confirmation AND (CHOP < 38.2 OR Price < Donchian Mid)
# - Short: Price < Donchian Low + volume confirmation AND (CHOP < 38.2 OR Price > Donchian Mid)
# - Exit: Price crosses Donchian midpoint
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volume confirms breakouts, chop filter avoids whipsaws in ranging markets
# - Target: 75-150 total trades over 4 years (19-37/year) for 4h timeframe

name = "4h_1d_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for volume MA
        return np.zeros(n)
    
    # Pre-compute 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channels (20-period) - based on completed bars only
    donchian_high = np.full(len(close), np.nan)
    donchian_low = np.full(len(close), np.nan)
    donchian_mid = np.full(len(close), np.nan)
    
    for i in range(20, len(close)):
        # Use previous 20 completed bars (0-19) to avoid look-ahead
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Calculate 4h Choppiness Index (14-period)
    chop = np.full(len(close), np.nan)
    for i in range(14, len(close)):
        # True Range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of True Range over 14 periods
        sum_tr = np.sum(tr[i-13:i+1])
        
        # Highest high and lowest low over 14 periods
        hh = np.max(high[i-13:i+1])
        ll = np.min(low[i-13:i+1])
        
        if hh > ll and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral when no range
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align HTF indicators to 4h timeframe
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Chop regime: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        chop_trending = chop[i] < 38.2
        chop_ranging = chop[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Donchian High + volume confirmation AND (trending OR (ranging AND price < mid))
            long_condition = (close[i] > donchian_high[i]) and volume_confirm and (chop_trending or (chop_ranging and close[i] < donchian_mid[i]))
            
            # Short entry: Price < Donchian Low + volume confirmation AND (trending OR (ranging AND price > mid))
            short_condition = (close[i] < donchian_low[i]) and volume_confirm and (chop_trending or (chop_ranging and close[i] > donchian_mid[i]))
            
            if long_condition:
                position = 1
                signals[i] = 0.25
            elif short_condition:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Donchian midpoint
            if position == 1:  # Long position
                if close[i] < donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals