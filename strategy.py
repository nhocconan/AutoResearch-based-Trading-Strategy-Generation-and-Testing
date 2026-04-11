#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on daily data
    # Jaw (blue): 13-period SMMA shifted 8 bars forward
    # Teeth (red): 8-period SMMA shifted 5 bars forward  
    # Lips (green): 5-period SMMA shifted 3 bars forward
    
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate SMMA for different periods
    close_1d = df_1d['close'].values
    jaw_raw = smma(close_1d, 13)  # Jaw: 13-period
    teeth_raw = smma(close_1d, 8)  # Teeth: 8-period
    lips_raw = smma(close_1d, 5)   # Lips: 5-period
    
    # Apply forward shift (Alligator shifts lines into future)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set NaN for shifted values that don't have data
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    
    # 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.006 * price_close  # ATR > 0.6% of price
        
        # Alligator conditions:
        # Green lips below red teeth below blue jaw = bearish alignment (mouth opening down)
        # Green lips above red teeth above blue jaw = bullish alignment (mouth opening up)
        lips_val = lips_6h[i]
        teeth_val = teeth_6h[i]
        jaw_val = jaw_6h[i]
        
        bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
        bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
        
        # Additional filter: price must be outside the Alligator's mouth
        # For longs: price above jaws (strong bullish)
        # For shorts: price below jaws (strong bearish)
        price_above_jaw = price_close > jaw_val
        price_below_jaw = price_close < jaw_val
        
        # Long conditions: bullish alignment + price above jaw + volume + volatility
        long_signal = bullish_aligned and price_above_jaw and volume_confirmed and vol_filter
        
        # Short conditions: bearish alignment + price below jaw + volume + volatility
        short_signal = bearish_aligned and price_below_jaw and volume_confirmed and vol_filter
        
        # Exit when Alligator lines re-align (mouth closes) or opposite signal
        exit_long = position == 1 and not bullish_aligned
        exit_short = position == -1 and not bearish_aligned
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25  # Size: 25%
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Williams Alligator on daily timeframe identifies strong trending regimes.
# Enters long when daily Alligator shows bullish alignment (lips>teeth>jaw) and 6h price
# is above the jaw (strong bullish momentum), with volume and volatility confirmation.
# Enters short when daily Alligator shows bearish alignment (lips<teeth<jaw) and 6h price
# is below the jaw (strong bearish momentum), with volume and volatility confirmation.
# Exits when the Alligator's mouth closes (lines re-intertwine), signaling trend exhaustion.
# Uses daily timeframe for trend identification to avoid 6h whipsaw, with 6h timing for entry.
# Works in both bull (catching strong uptrends) and bear (catching strong downtrends) markets.