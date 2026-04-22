#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index with 1w RSI filter and volume confirmation.
# Choppiness Index > 61.8 indicates ranging market (mean reversion opportunity).
# Combined with 1w RSI < 40 for long or > 60 for short, and volume spikes (>1.5x 20-period avg),
# this captures mean reversion moves in ranging markets while avoiding strong trends.
# Designed for low trade frequency (~15-25/year) to minimize fee decay.
# Works in both bull and bear markets by adapting to ranging conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for RSI filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 14-period RSI on weekly close
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    
    # Align weekly RSI to 12h timeframe (waits for weekly bar to close)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 14-period Choppiness Index on 12h high/low/close
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula: 100 * log10(TR_sum / (HH - LL)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # small value to prevent div by zero
    chop = 100 * np.log10(tr_sum / hl_range) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe (no additional delay needed)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_aligned[i]
        rsi_val = rsi_1w_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Choppiness conditions: > 61.8 = ranging (mean reversion opportunity)
        ranging = chop_val > 61.8
        
        if position == 0:
            # Long conditions: ranging market + RSI oversold + price near low + volume spike
            if ranging and rsi_val < 40 and price <= ll[i] * 1.02 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: ranging market + RSI overbought + price near high + volume spike
            elif ranging and rsi_val > 60 and price >= hh[i] * 0.98 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to neutral or chop drops (trending begins) or price reaches high
                if rsi_val >= 50 or chop_val < 50 or price >= hh[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to neutral or chop drops (trending begins) or price reaches low
                if rsi_val <= 50 or chop_val < 50 or price <= ll[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Choppiness_RSI_Volume_MeanReversion"
timeframe = "12h"
leverage = 1.0