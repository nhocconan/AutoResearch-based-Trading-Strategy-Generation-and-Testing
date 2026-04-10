#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w/1d regime filter and volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1w close > 1w open (bullish weekly) AND volume > 1.5x 20-period average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1w close < 1w open (bearish weekly) AND volume > 1.5x 20-period average
# - Exit when Alligator alignment breaks (Lips crosses Teeth or Jaw) OR price crosses back below/above Lips
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Alligator identifies trending vs ranging markets; only trade in strong trends
# - Weekly regime filter ensures alignment with higher timeframe momentum
# - Volume confirmation reduces false signals

name = "12h_1w_1d_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 13 or len(df_1d) < 13:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator (typical price)
    median_price = (high + low + close) / 3.0
    
    # Williams Alligator SMAs (using SMA, not EMA, as per original)
    def sma(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    # Jaw: 13-period SMA, smoothed by 8 bars
    jaw_raw = sma(median_price, 13)
    jaw = sma(jaw_raw, 8) if len(jaw_raw) >= 8 else np.full_like(jaw_raw, np.nan)
    
    # Teeth: 8-period SMA, smoothed by 5 bars
    teeth_raw = sma(median_price, 8)
    teeth = sma(teeth_raw, 5) if len(teeth_raw) >= 5 else np.full_like(teeth_raw, np.nan)
    
    # Lips: 5-period SMA, smoothed by 3 bars
    lips_raw = sma(median_price, 5)
    lips = sma(lips_raw, 3) if len(lips_raw) >= 3 else np.full_like(lips_raw, np.nan)
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w bullish/bearish regime (close > open = bullish)
    weekly_bullish = df_1w['close'].values > df_1w['open'].values
    weekly_bearish = df_1w['close'].values < df_1w['open'].values
    
    # Pre-compute 1d bullish/bearish regime (close > open = bullish)
    daily_bullish = df_1d['close'].values > df_1d['open'].values
    daily_bearish = df_1d['close'].values < df_1d['open'].values
    
    # Align HTF indicators to 12h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(daily_bullish_aligned[i]) or 
            np.isnan(daily_bearish_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bullish alignment AND price > Lips AND weekly bullish AND daily bullish AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and  # Bullish alignment
                close[i] > lips[i] and 
                weekly_bullish_aligned[i] and 
                daily_bullish_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish alignment AND price < Lips AND weekly bearish AND daily bearish AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and  # Bearish alignment
                  close[i] < lips[i] and 
                  weekly_bearish_aligned[i] and 
                  daily_bearish_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator alignment breaks OR price crosses back below/above Lips
            exit_long = False
            exit_short = False
            
            if position == 1:  # Long position
                # Exit if alignment breaks (not bullish) OR price <= Lips
                if not (lips[i] > teeth[i] and teeth[i] > jaw[i]):  # Alignment broken
                    exit_long = True
                elif close[i] <= lips[i]:  # Price back below Lips
                    exit_long = True
            else:  # Short position
                # Exit if alignment breaks (not bearish) OR price >= Lips
                if not (lips[i] < teeth[i] and teeth[i] < jaw[i]):  # Alignment broken
                    exit_short = True
                elif close[i] >= lips[i]:  # Price back above Lips
                    exit_short = True
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals