#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with volume confirmation and ATR stoploss
# Uses 20-day Donchian channels on daily timeframe for breakout signals,
# confirmed by daily volume > 1.5x 20-day average volume.
# Includes ATR-based stoploss and volatility-adjusted position sizing.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns.

name = "daily_donchian20_volume_vol_scaled_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian(20) - upper and lower bands
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    # ATR(20) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 0=flat, 1=long, -1=short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: scale position size based on volatility
        vol_ratio_atr = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        vol_scale = np.clip(1.0 / vol_ratio_atr, 0.5, 1.5)  # scale between 0.5 and 1.5
        base_size = 0.25
        
        # Breakout conditions with volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        # Long breakout: price breaks above upper Donchian band with volume
        long_breakout = (close[i] > highest_high[i]) and vol_confirmed
        
        # Short breakdown: price breaks below lower Donchian band with volume
        short_breakout = (close[i] < lowest_low[i]) and vol_confirmed
        
        # Stoploss: exit if price moves against position by 2*ATR
        if position == 1 and i > 0:
            # Track entry price approximation (using close of entry bar)
            if signals[i-1] == 0 and signals[i-1] != position:
                entry_price = close[i-1]  # approximate entry at previous close
            else:
                # Find actual entry price by looking back
                j = i-1
                while j > 0 and signals[j] == position:
                    j -= 1
                if j >= 0 and signals[j] == 0 and signals[j+1] == position:
                    entry_price = close[j+1]
                else:
                    entry_price = close[i-1]  # fallback
            
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1 and i > 0:
            # Track entry price approximation
            if signals[i-1] == 0 and signals[i-1] != position:
                entry_price = close[i-1]
            else:
                j = i-1
                while j > 0 and signals[j] == position:
                    j -= 1
                if j >= 0 and signals[j] == 0 and signals[j+1] == position:
                    entry_price = close[j+1]
                else:
                    entry_price = close[i-1]
            
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Generate new signals
        if long_breakout and position != 1:
            signals[i] = base_size * vol_scale
            position = 1
        elif short_breakout and position != -1:
            signals[i] = -base_size * vol_scale
            position = -1
        else:
            # Hold current position
            signals[i] = base_size * vol_scale if position == 1 else (-base_size * vol_scale if position == -1 else 0.0)
    
    return signals