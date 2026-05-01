#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d EMA34 trend + volume spike
# Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA34 AND volume > 1.5x 20-period 6h volume average
# Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA34 AND volume > 1.5x 20-period 6h volume average
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Williams %R calculated from prior completed 6h bar to avoid look-ahead.
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend) regimes.
# Target: 12-30 trades/year on 6h timeframe.

name = "6h_WilliamsR_Extreme_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and Williams %R
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Williams %R(14): (highest high - close) / (highest high - lowest low) * -100
        # Using prior completed bars to avoid look-ahead
        if i < 14:
            signals[i] = 0.0
            continue
            
        highest_high = np.max(high[i-14:i])  # excludes current bar
        lowest_low = np.min(low[i-14:i])     # excludes current bar
        
        if highest_high == lowest_low:  # avoid division by zero
            williams_r = -50.0
        else:
            williams_r = ((highest_high - curr_close) / (highest_high - lowest_low)) * -100
        
        # Volume spike: current volume > 1.5x 20-period 6h volume average
        if i < 20:
            vol_ma = np.mean(volume[max(0, i-20):i])  # use available history
        else:
            vol_ma = np.mean(volume[i-20:i])
        
        if vol_ma <= 0:
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma * 1.5)
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) AND volume spike AND uptrend
            if (williams_r < -80 and 
                volume_spike and 
                uptrend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R overbought (> -20) AND volume spike AND downtrend
            elif (williams_r > -20 and 
                  volume_spike and 
                  downtrend):
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
            # Exit: Williams %R returns to neutral range OR trend reverses
            elif (williams_r > -50) or (curr_close < ema_34_1d_aligned[i]):  # trend reversal
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
            # Exit: Williams %R returns to neutral range OR trend reverses
            elif (williams_r < -50) or (curr_close > ema_34_1d_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals