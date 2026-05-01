#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend + volume spike + choppiness regime filter.
# Long when price breaks above Donchian upper AND 1d EMA34 rising AND volume > 2x 20-bar avg AND CHOP > 61.8 (ranging market for mean reversion bounce).
# Short when price breaks below Donchian lower AND 1d EMA34 falling AND volume > 2x 20-bar avg AND CHOP > 61.8.
# Uses discrete sizing 0.25 to minimize fee churn. Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) via 1d EMA34 slope filter.
# Choppiness regime filter avoids whipsaws in strong trends by only trading range-bound markets where breakouts are more likely to fail and revert.

name = "4h_Donchian20_1dEMA34_VolumeSpike_ChopRegime_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d EMA34 slope (rising/falling)
    ema_34_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_34_rising = ema_34_slope > 0
    ema_34_falling = ema_34_slope < 0
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 4h volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) - ranges 0-100, >61.8 = ranging, <38.2 = trending
    def calculate_chop(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.close_shift)), np.abs(low - np.close_shift))).rolling(window).sum()
        max_high = pd.Series(high).rolling(window).max()
        min_low = pd.Series(low).rolling(window).min()
        chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(window)
        return chop.values
    
    # Shift close for true range calculation
    close_shift = np.roll(close, 1)
    close_shift[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - close_shift)
    tr3 = np.abs(low - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 4h timeframe
        hour = hours[i]
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        chop_filter = chop[i] > 61.8  # only trade in ranging markets
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_upper[i-1]  # break above previous upper band
        breakout_down = curr_low < donchian_lower[i-1]  # break below previous lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian upper AND 1d EMA34 rising AND volume confirmation AND chop filter
            if (breakout_up and 
                ema_34_rising[i] and 
                volume_confirm and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian lower AND 1d EMA34 falling AND volume confirmation AND chop filter
            elif (breakout_down and 
                  ema_34_falling[i] and 
                  volume_confirm and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian lower (stoploss) OR 1d EMA34 falls (trend change) OR chop breaks below 38.2 (trending market)
            if (curr_low < donchian_lower[i] or 
                ema_34_falling[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper (stoploss) OR 1d EMA34 rises (trend change) OR chop breaks below 38.2 (trending market)
            if (curr_high > donchian_upper[i] or 
                ema_34_rising[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals