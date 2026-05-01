#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation + choppiness regime filter.
# Uses 1d for signal direction (trend + chop regime), 4h only for entry timing precision.
# Long when price breaks above Donchian upper band with 1d EMA34 uptrend, CHOP > 61.8 (range) and volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band with 1d EMA34 downtrend, CHOP > 61.8 (range) and volume confirmation.
# Discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Target: 75-150 total trades over 4 years (19-37/year) to balance edge and fee drag.

name = "4h_Donchian_20_1dEMA34_Trend_VolumeChop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate choppiness index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (log10(HHLL))) / log10(14)
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_1d = np.concatenate([[np.max([df_1d['high'].values[0] - df_1d['low'].values[0], 
                                     np.abs(df_1d['high'].values[0] - df_1d['close'].values[0]), 
                                     np.abs(df_1d['low'].values[0] - df_1d['close'].values[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    hhll = hh - ll
    chop = 100 * np.log10(sum_atr14 / hhll) / np.log10(14)
    chop = np.where(hhll == 0, 50, chop)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels from 1d OHLC (using previous day's OHLC to avoid look-ahead)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of last 20 periods
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 34  # warmup for EMA34, ATR and choppiness
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.5)
        
        # Regime filter: only trade in choppy markets (CHOP > 61.8 = range)
        regime_filter = chop_aligned[i] > 61.8
        
        # Donchian breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = curr_close > upper_band_aligned[i-1]  # break above previous upper band
        breakout_down = curr_close < lower_band_aligned[i-1]  # break below previous lower band
        
        # Trend filter: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = curr_close > ema_34_aligned[i]
        bearish_trend = curr_close < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND bullish trend AND volume confirmation AND chop regime
            if (breakout_up and 
                bullish_trend and 
                volume_confirm and 
                regime_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND bearish trend AND volume confirmation AND chop regime
            elif (breakout_down and 
                  bearish_trend and 
                  volume_confirm and 
                  regime_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR trend turns bearish OR chop breaks down (trending)
            elif (curr_close < upper_band_aligned[i] and curr_close > lower_band_aligned[i]) or \
                 bearish_trend or \
                 chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR trend turns bullish OR chop breaks down (trending)
            elif (curr_close < upper_band_aligned[i] and curr_close > lower_band_aligned[i]) or \
                 bullish_trend or \
                 chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals