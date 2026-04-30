#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR(14) volume confirmation
# Donchian breakouts capture strong momentum moves; 1d EMA34 ensures alignment with daily trend
# Volume confirmation requires current volume > 1.5x ATR-scaled average volume to filter low-conviction moves
# ATR stoploss exits when price moves against position by 2.5x ATR(14) to manage risk
# Discrete sizing 0.30 balances return and drawdown. Target: 100-200 total trades over 4 years (25-50/year).
# Works in bull markets via upside breakouts and bear markets via downside breaks with trend filter.

name = "4h_Donchian20_1dEMA34_ATRVol_v1"
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
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for 4h timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR-scaled average volume (ATR * 100 as proxy for dollar volume)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    atr_scaled_vol = atr_14 * 100
    vol_ma_50_scaled = pd.Series(atr_scaled_vol).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.5 * vol_ma_50_scaled / 100)  # Convert back to volume units
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_50[i]) or np.isnan(vol_ma_50_scaled[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr_14[i]
        curr_highest = highest_20[i]
        curr_lowest = lowest_20[i]
        curr_vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume confirmation and trend alignment
            if curr_vol_confirm:
                # Bullish entry: break above upper Donchian with close > 1d EMA34
                if curr_close > curr_highest and curr_close > curr_ema:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below lower Donchian with close < 1d EMA34
                elif curr_close < curr_lowest and curr_close < curr_ema:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # ATR trailing stop: exit when price drops below highest close - 2.5*ATR
            # or when price breaks below lower Donchian (failed breakout)
            trailing_stop = entry_price - 2.5 * curr_atr
            if curr_close < trailing_stop or curr_close < curr_lowest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # ATR trailing stop: exit when price rises above lowest close + 2.5*ATR
            # or when price breaks above upper Donchian (failed breakdown)
            trailing_stop = entry_price + 2.5 * curr_atr
            if curr_close > trailing_stop or curr_close > curr_highest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals