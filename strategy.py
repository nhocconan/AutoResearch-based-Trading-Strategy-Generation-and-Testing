#!/usr/bin/env python3
"""
12h_donchian_20_1w_trend_volume_v2
Hypothesis: On 12-hour timeframe, use weekly Donchian channel breakout for trend direction, combined with weekly EMA filter and volume confirmation.
Enter long when price breaks above weekly Donchian high (20-period) AND price > weekly EMA20 AND 12h volume > 1.5x 20-period average.
Enter short when price breaks below weekly Donchian low (20-period) AND price < weekly EMA20 AND 12h volume > 1.5x 20-period average.
Exit when price returns to weekly EMA20 or volume drops below average.
This captures strong trends with institutional participation while avoiding whipsaw in choppy markets.
Target: 15-25 trades/year to minimize fee drag while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_1w_trend_volume_v2"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20-period)
    period = 20
    w_high_series = pd.Series(w_high)
    w_low_series = pd.Series(w_low)
    donch_high = w_high_series.rolling(window=period, min_periods=period).max().values
    donch_low = w_low_series.rolling(window=period, min_periods=period).min().values
    
    # Calculate weekly EMA (20-period)
    w_close_series = pd.Series(w_close)
    w_ema = w_close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    w_ema_aligned = align_htf_to_ltf(prices, df_1w, w_ema)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if weekly data not available
        if np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or np.isnan(w_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Weekly breakout conditions
        breakout_high = close[i] > donch_high_aligned[i]
        breakout_low = close[i] < donch_low_aligned[i]
        
        # Weekly EMA filter
        price_above_ema = close[i] > w_ema_aligned[i]
        price_below_ema = close[i] < w_ema_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price returns to weekly EMA
            if price_below_ema:
                exit_long = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price returns to weekly EMA
            if price_above_ema:
                exit_short = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high AND price above weekly EMA AND volume confirmed
            long_entry = breakout_high and price_above_ema and vol_confirmed
            
            # Short entry: breakout below Donchian low AND price below weekly EMA AND volume confirmed
            short_entry = breakout_low and price_below_ema and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals