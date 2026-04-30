#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume spike + ADX regime filter.
# Long when price breaks above Donchian upper band AND volume > 1.5x 20-bar avg AND ADX > 25 (trending).
# Short when price breaks below Donchian lower band AND volume > 1.5x 20-bar avg AND ADX > 25 (trending).
# Exit when price crosses Donchian midline (mean of upper/lower band) OR ATR stoploss (2.0x ATR).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Donchian provides structural breakouts; volume confirms conviction; ADX filters for trending markets only.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull/bear via ADX trend filter.

name = "4h_Donchian20_VolumeSpike_ADXTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # Calculate ADX for regime filter (trending market detection)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14)  # warmup for Donchian, ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian upper band, trending market (ADX > 25), volume confirmation
            if (curr_high > highest_20[i] and 
                adx[i] > 25 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Donchian lower band, trending market (ADX > 25), volume confirmation
            elif (curr_low < lowest_20[i] and 
                  adx[i] > 25 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: Donchian midline cross OR ATR stoploss
            exit_signal = False
            if curr_close < donchian_mid[i]:  # midline cross
                exit_signal = True
            elif curr_close < entry_price - 2.0 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Donchian midline cross OR ATR stoploss
            exit_signal = False
            if curr_close > donchian_mid[i]:  # midline cross
                exit_signal = True
            elif curr_close > entry_price + 2.0 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals