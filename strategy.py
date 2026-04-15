#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d RSI mean reversion
# Uses Choppiness Index (14) on 12h to identify ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In ranging markets: RSI(14) on 1d for mean reversion (long RSI<30, short RSI>70).
# In trending markets: 1d EMA(50) trend filter (long price>EMA, short price<EMA).
# Volume confirmation: current volume > 1.5x 20-period median volume.
# Designed for low trade frequency (<30/year) to avoid fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for RSI and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)  # Avoid division by zero
    
    # Align Choppiness Index to 12h timeframe (no extra delay needed)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate RSI (14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate EMA(50) on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            continue
        
        chop_val = chop_aligned[i]
        rsi_val = rsi_aligned[i]
        close_price = close[i]
        ema_price = ema_50_aligned[i]
        
        # Volume confirmation: current volume > 1.5x median of last 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        vol_confirmed = volume[i] > 1.5 * vol_median
        
        # Determine market regime
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        # Long entry conditions
        long_entry = False
        if is_ranging and rsi_val < 30 and vol_confirmed:
            long_entry = True  # Mean reversion in ranging market
        elif is_trending and close_price > ema_price and vol_confirmed:
            long_entry = True  # Trend following in trending market
        
        # Short entry conditions
        short_entry = False
        if is_ranging and rsi_val > 70 and vol_confirmed:
            short_entry = True  # Mean reversion in ranging market
        elif is_trending and close_price < ema_price and vol_confirmed:
            short_entry = True  # Trend following in trending market
        
        # Exit conditions: reverse signal or regime change to extreme chop
        exit_long = position == 1 and (
            (is_ranging and rsi_val > 70) or  # Opposite RSI in ranging
            (is_trending and close_price < ema_price) or  # Trend reversal
            chop_val > 80  # Extreme chop (avoid whipsaw)
        )
        
        exit_short = position == -1 and (
            (is_ranging and rsi_val < 30) or  # Opposite RSI in ranging
            (is_trending and close_price > ema_price) or  # Trend reversal
            chop_val > 80  # Extreme chop
        )
        
        # Generate signals
        if long_entry and position <= 0:
            position = 1
            signals[i] = base_size
        elif short_entry and position >= 0:
            position = -1
            signals[i] = -base_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Chop_RSI_EMA_Regime"
timeframe = "12h"
leverage = 1.0