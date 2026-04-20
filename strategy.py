#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1-day volume spike filter and ATR-based stoploss
# In trending markets: breakout above/below 20-period Donchian channel + volume > 1.5x average
# In ranging markets: fade at Donchian channel extremes when RSI(14) shows divergence
# Uses 1-day volume for confirmation and ATR(14) for dynamic position sizing and stoploss
# Designed to capture trends while avoiding false breakouts in low-volume conditions
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate average daily volume (20-period SMA)
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 12-period RSI for divergence detection
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR(14) for stoploss and position sizing
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical indicators
        if (np.isnan(avg_vol_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].values[i]
        
        # Volume condition: current volume > 1.5x average daily volume
        vol_condition = volume > (1.5 * avg_vol_1d_aligned[i])
        
        # Donchian breakout conditions
        upper_break = price > donchian_high[i]
        lower_break = price < donchian_low[i]
        
        # RSI divergence conditions for fading
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        if position == 0:
            # Enter long: Donchian breakout up + volume confirmation
            if upper_break and vol_condition:
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakout down + volume confirmation
            elif lower_break and vol_condition:
                signals[i] = -0.25
                position = -1
            # Fade extremes in low volume conditions with RSI divergence
            elif not vol_condition:
                if rsi_overbought and price >= donchian_high[i] * 0.995:  # Near upper band
                    signals[i] = -0.25
                    position = -1
                elif rsi_oversold and price <= donchian_low[i] * 1.005:  # Near lower band
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Exit long: Donchian breakdown or RSI overbought
            if price < donchian_low[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout up or RSI oversold
            if price > donchian_high[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_RSIFade"
timeframe = "12h"
leverage = 1.0