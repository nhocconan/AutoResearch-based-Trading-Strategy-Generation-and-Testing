#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_williams_alligator_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return signals
    
    # Williams Alligator components (13, 8, 5 period SMAs with future shifts)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, 8 bars ahead
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values    # 8-period, 5 bars ahead
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values     # 5-period, 3 bars ahead
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX filter for trending markets
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    trending_market = adx > 25
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx[i]
        trending = trending_market[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Williams Alligator signals: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
        uptrend = lips_val > teeth_val and teeth_val > jaw_val
        downtrend = lips_val < teeth_val and teeth_val < jaw_val
        
        # Entry signals - only in trending markets with volume
        long_signal = False
        short_signal = False
        
        # Long: price above lips in uptrend with volume confirmation
        if price_close > lips_val and uptrend and volume_confirmed and trending:
            long_signal = True
        
        # Short: price below lips in downtrend with volume confirmation
        if price_close < lips_val and downtrend and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions
        # Stop loss based on ATR
        stop_long = position == 1 and price_low < (entry_price - 2.5 * atr[i])
        stop_short = position == -1 and price_high > (entry_price + 2.5 * atr[i])
        
        # Exit when Alligator closes (lips crosses teeth in opposite direction)
        exit_long = position == 1 and lips_val < teeth_val
        exit_short = position == -1 and lips_val > teeth_val
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Williams Alligator strategy on 12h timeframe with daily Alligator alignment.
# Enters long when price is above lips and Alligator is aligned bullish (lips>teeth>jaw) with volume confirmation.
# Enters short when price is below lips and Alligator is aligned bearish (lips<teeth<jaw) with volume confirmation.
# Uses daily timeframe for Alligator calculation to avoid noise and capture multi-day trends.
# Volume confirmation ensures institutional participation, ADX filter (>25) avoids whipsaws.
# Exits when Alligator closes (lips crosses teeth) or ATR stop loss (2.5x) is hit.
# Designed for 12h timeframe to target 50-150 total trades over 4 years.
# Works in both bull and bear markets by trading with the trend as defined by the Alligator.