#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above 20-period high, volume > 1.5x 20-bar average, and ATR(14) < 0.05 * price (low volatility filter).
# Short when price breaks below 20-period low, volume > 1.5x 20-bar average, and ATR(14) < 0.05 * price.
# Exit via ATR trailing stop: long exits when price < highest high since entry - 2.5 * ATR, short exits when price > lowest low since entry + 2.5 * ATR.
# Uses 12h timeframe to target 20-50 trades/year. Volume confirmation reduces false breakouts, ATR stop manages risk.
# Works in bull markets via breakouts and in bear markets via breakdowns with volatility filter to avoid chop.

name = "12h_Donchian20_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels: 20-period high and low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Low volatility filter: ATR < 5% of price (avoid choppy markets)
    vol_filter = (atr / close) < 0.05
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    start_idx = 50  # warmup for Donchian, ATR, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirm[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_vol_filter = vol_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, volume confirmation, low volatility
            if (curr_close > curr_donchian_high and 
                curr_volume_confirm and 
                curr_vol_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high = curr_high
            # Short: price breaks below Donchian low, volume confirmation, low volatility
            elif (curr_close < curr_donchian_low and 
                  curr_volume_confirm and 
                  curr_vol_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_low = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_high = max(highest_high, curr_high)
            # ATR trailing stop: exit when price < highest high - 2.5 * ATR
            if curr_close < (highest_high - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low = min(lowest_low, curr_low)
            # ATR trailing stop: exit when price > lowest low + 2.5 * ATR
            if curr_close > (lowest_low + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals