#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly (Monday) open gap fill with volume confirmation and trend filter.
# Weekly gaps (price opens outside previous week's range) often revert to fill the gap.
# We go long when price opens below prior week's low and short when above prior week's high,
# only if volume confirms (>1.5x 20-day average) and price is above/below 50-week EMA for trend alignment.
# Designed for very low frequency (~5-15 trades/year) to minimize fee decay.
# Works in both bull and bear markets by using weekly structure and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for gap calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_open = df_weekly['open'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate 50-week EMA for trend filter
    ema_50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    weekly_open_aligned = align_htf_to_ltf(prices, df_weekly, weekly_open)
    ema_50_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate 20-day average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_open_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current day's open price (first available price of the day)
        # Since we're using daily timeframe, we use the open price
        open_price = prices['open'].iloc[i]
        close_price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        weekly_high_val = weekly_high_aligned[i]
        weekly_low_val = weekly_low_aligned[i]
        weekly_open_val = weekly_open_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_confirmed = vol > 1.5 * vol_ma
        
        # Gap conditions: open outside prior week's range
        gap_down = open_price < weekly_low_val  # opened below weekly low
        gap_up = open_price > weekly_high_val   # opened above weekly high
        
        if position == 0:
            # Long: gap down + volume confirmation + price above weekly EMA (bullish bias)
            if gap_down and vol_confirmed and open_price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: gap up + volume confirmation + price below weekly EMA (bearish bias)
            elif gap_up and vol_confirmed and open_price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite side of week's range or trend breaks
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to or above weekly high (gap filled) or trend breaks
                if close_price >= weekly_high_val or close_price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to or below weekly low (gap filled) or trend breaks
                if close_price <= weekly_low_val or close_price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyGapFill_VolumeTrend"
timeframe = "1d"
leverage = 1.0