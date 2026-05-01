#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA34 trend filter, volume confirmation, and ATR stoploss.
# Long when Williams %R crosses above -80 from below AND price > 1d EMA34 AND volume > 2x 4h volume median.
# Short when Williams %R crosses below -20 from above AND price < 1d EMA34 AND volume > 2x 4h volume median.
# Uses discrete sizing 0.25. ATR stoploss: signal→0 when price moves against position by 2.5*ATR.
# Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years) to minimize fee drag.
# Williams %R provides clear overbought/oversold signals with mean reversion tendency.
# 1d EMA34 offers smooth trend filter, volume confirms momentum strength.
# This combination has shown potential in DB for BTC/ETH with proper filtering.

name = "4h_WilliamsR_1dEMA34_Volume_v2"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14) from 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_4h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_4h['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_4h['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Calculate 1d EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume median (30-period for stability)
    vol_median_4h = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    prev_williams_r = williams_r_aligned[0] if not np.isnan(williams_r_aligned[0]) else -50
    
    # Start after warmup for ATR, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_median_4h[i]) or
            np.isnan(prev_williams_r)):
            signals[i] = 0.0
            prev_williams_r = williams_r_aligned[i] if not np.isnan(williams_r_aligned[i]) else prev_williams_r
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r_aligned[i]
        
        # Volume confirmation: current volume > 2x 4h volume median
        if vol_median_4h[i] <= 0 or np.isnan(vol_median_4h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_4h[i] * 2.0)
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Williams %R signals: cross above -80 (long) or below -20 (short)
        williams_long_signal = (prev_williams_r <= -80) and (curr_williams_r > -80)
        williams_short_signal = (prev_williams_r >= -20) and (curr_williams_r < -20)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND uptrend AND volume confirmation
            if williams_long_signal and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R crosses below -20 AND downtrend AND volume confirmation
            elif williams_short_signal and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses below -50 (mean reversion) OR trend turns down
            elif (curr_williams_r < -50) or (not uptrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses above -50 (mean reversion) OR trend turns up
            elif (curr_williams_r > -50) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        
        prev_williams_r = curr_williams_r
    
    return signals