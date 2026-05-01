#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# Long when Alligator Jaw (13-period SMMA) < Teeth (8-period SMMA) < Lips (5-period SMMA) AND 1w close > 1w EMA34 AND volume > 2.0x 50-period median.
# Short when Alligator Jaw > Teeth > Lips AND 1w close < 1w EMA34 AND volume > 2.0x 50-period median.
# The Alligator identifies trend phases; 1w EMA34 provides higher-timeframe trend alignment; volume confirms conviction.
# Works in bull markets (buy on bullish alignment in uptrend) and bear markets (sell on bearish alignment in downtrend).
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to minimize fee drag.

name = "1d_WilliamsAlligator_1wEMA34_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator: SMMA (Smoothed Moving Average) = EMA with alpha=1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)   # Jaw (13-period SMMA)
    teeth = smma(close, 8)  # Teeth (8-period SMMA)
    lips = smma(close, 5)   # Lips (5-period SMMA)
    
    # Calculate 50-period volume median for volume confirmation
    vol_median_50 = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for SMMA and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_median_50[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA34 direction
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 50-period volume median
        if vol_median_50[i] <= 0 or np.isnan(vol_median_50[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_50[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Jaw < Teeth < Lips
            bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            # Bearish alignment: Jaw > Teeth > Lips
            bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Long: Bullish alignment AND uptrend AND volume spike
            if bullish_alignment and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish alignment AND downtrend AND volume spike
            elif bearish_alignment and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bullish alignment breaks OR trend turns down
            bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            if not bullish_alignment or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bearish alignment breaks OR trend turns up
            bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            if not bearish_alignment or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals