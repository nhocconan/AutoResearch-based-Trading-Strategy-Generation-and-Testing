#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and chop regime filter
# - Long when price breaks above Donchian(20) high + ATR(14) > ATR(50) (expanding volatility) + Chop < 38.2 (trending regime)
# - Short when price breaks below Donchian(20) low + ATR(14) > ATR(50) + Chop < 38.2
# - Exit: price returns to Donchian midpoint (mean reversion within channel)
# - Position sizing: 0.25 discrete level
# - Donchian provides clear trend structure, ATR filter ensures volatility expansion, chop filter avoids ranging markets
# - Works in bull/bear: breakouts capture strong moves, chop filter avoids false signals in ranges
# - 4h timeframe targets 25-60 trades/year with strict entry conditions to minimize fee drag

name = "4h_1d_donchian_atr_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4h Chopiness Index (14-period) for regime filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Sum of TR over period
    sum_tr = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop formula: 100 * log10(sum_TR / (HH - LL)) / log10(N)
    hl_range = hh - ll
    chop = np.where((hl_range > 0) & (sum_tr > 0), 
                    100 * np.log10(sum_tr / hl_range) / np.log10(14), 
                    50)  # default to neutral when invalid
    
    # Calculate 4h ATR(14) and ATR(50) for volatility filter
    atr14 = np.maximum(high - low, 
                       np.maximum(np.abs(high - np.roll(close, 1)), 
                                  np.abs(low - np.roll(close, 1))))
    atr14[0] = high[0] - low[0]
    atr14_ma = pd.Series(atr14).rolling(window=14, min_periods=14).mean().values
    
    atr50 = np.maximum(high - low, 
                       np.maximum(np.abs(high - np.roll(close, 1)), 
                                  np.abs(low - np.roll(close, 1))))
    atr50[0] = high[0] - low[0]
    atr50_ma = pd.Series(atr50).rolling(window=50, min_periods=50).mean().values
    
    # Volatility expansion: ATR(14) > ATR(50) * 1.1
    vol_expansion = atr14_ma > (atr50_ma * 1.1)
    
    # Calculate 1d ATR for HTF volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_expansion[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chop < 38.2 indicates trending market (favorable for breakouts)
        trending_market = chop[i] < 38.2
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i]   # Price breaks above upper band
        breakout_down = close[i] < donchian_low[i]  # Price breaks below lower band
        return_to_mid = abs(close[i] - donchian_mid[i]) < (donchian_high[i] - donchian_low[i]) * 0.2  # Within 20% of midpoint
        
        # Entry conditions: Donchian breakout with volatility expansion and trending regime
        long_entry = breakout_up and vol_expansion[i] and trending_market
        short_entry = breakout_down and vol_expansion[i] and trending_market
        
        # Exit conditions: price returns to Donchian midpoint (mean reversion within channel)
        long_exit = return_to_mid  # Exit long when price returns to midpoint
        short_exit = return_to_mid  # Exit short when price returns to midpoint
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals