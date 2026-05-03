#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based stoploss.
# Uses actual Donchian channel calculation from prior 4h bar (H, L) to derive upper/lower bands.
# Long when price breaks above upper band with volume > 1.3x 20-period MA and close > 1d EMA34 (uptrend).
# Short when price breaks below lower band with volume spike and close < 1d EMA34 (downtrend).
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide structural breakout levels; EMA34 filters counter-trend trades.
# Volume confirmation reduces false breakouts. ATR stoploss manages risk. Works in bull/bear via trend alignment.

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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) from prior 4h bar (H20, L20) - wait for completed bar
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Need to shift by 1 to use completed 4h bar only (no look-ahead)
    h20 = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    l20 = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 4h timeframe (wait for completed 4h bar)
    upper_band = align_htf_to_ltf(prices, df_4h, h20)
    lower_band = align_htf_to_ltf(prices, df_4h, l20)
    
    # Volume regime: current 4h volume > 1.3x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        upper = upper_band[i]
        lower = lower_band[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above upper band with volume spike in uptrend
            if close_val > upper and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: break below lower band with volume spike in downtrend
            elif close_val < lower and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: ATR-based stoploss OR price breaks below lower band OR trend turns down
            if close_val < entry_price - 2.0 * atr_val or close_val < lower or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ATR-based stoploss OR price breaks above upper band OR trend turns up
            if close_val > entry_price + 2.0 * atr_val or close_val > upper or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals