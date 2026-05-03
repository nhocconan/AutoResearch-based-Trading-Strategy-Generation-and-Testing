#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend + volume spike + ATR(14) stoploss.
# Long when price breaks above Donchian upper in bull trend (close > 1d EMA34) with volume > 1.5x 20-period MA.
# Short when price breaks below Donchian lower in bear trend (close < 1d EMA34) with volume spike.
# Donchian provides clear structure; 1d EMA34 filters whipsaw; volume confirms participation.
# ATR stoploss limits downside. Target: 100-200 total trades over 4 years (25-50/year) with discrete sizing 0.30.

name = "4h_Donchian20_1dEMA34_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # first value NaN
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            if is_bull_trend and close_val > upper and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif is_bear_trend and close_val < lower and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below lower band OR ATR stoploss hit OR trend reversal
            if (close_val < lower or 
                close_val < entry_price - 2.0 * atr_val or 
                close_val < ema_trend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above upper band OR ATR stoploss hit OR trend reversal
            if (close_val > upper or 
                close_val > entry_price + 2.0 * atr_val or 
                close_val > ema_trend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
    
    return signals