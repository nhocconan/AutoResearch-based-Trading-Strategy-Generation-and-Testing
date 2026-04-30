#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter, volume confirmation, and chop regime filter
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 40-80 total trades over 4 years (10-20/year).
# Long when price breaks above Donchian(20) high AND 1w EMA34 uptrend AND volume spike AND chop < 61.8
# Short when price breaks below Donchian(20) low AND 1w EMA34 downtrend AND volume spike AND chop < 61.8
# Exit on opposite Donchian(10) break or trend reversal. Designed to capture trends while avoiding whipsaws in ranging markets.

name = "1d_DonchianBreakout_1wEMA34_VolumeChop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 1w EMA(34) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index regime filter: avoid ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (np.log10(14) * (hh - ll) + 1e-10))
    chop_filter = chop < 61.8  # Trending market (not choppy)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 34, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high_20 = donchian_high_20[i]
        curr_donchian_low_20 = donchian_low_20[i]
        curr_donchian_high_10 = donchian_high_10[i]
        curr_donchian_low_10 = donchian_low_10[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_chop_filter = chop_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market (not choppy)
            if curr_volume_spike and curr_chop_filter:
                # Bullish entry: price breaks above Donchian(20) high AND above weekly EMA34
                if (curr_close > curr_donchian_high_20 and 
                    curr_close > curr_ema_34_1w):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian(20) low AND below weekly EMA34
                elif (curr_close < curr_donchian_low_20 and 
                      curr_close < curr_ema_34_1w):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian(10) low OR loses weekly uptrend
            if (curr_close < curr_donchian_low_10 or 
                curr_close < curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian(10) high OR loses weekly downtrend
            if (curr_close > curr_donchian_high_10 or 
                curr_close > curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals