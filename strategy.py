#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND close > 1d EMA50 AND volume > 2.0x 6h volume median.
# Short when Williams %R > -20 (overbought) AND close < 1d EMA50 AND volume > 2.0x 6h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Williams %R identifies exhaustion points; 1d EMA50 filters for longer-term trend alignment.
# Volume confirmation ensures momentum behind the move. Target: 12-25 trades/year on 6h timeframe.
# This combination avoids the saturated Camarilla/Donchian families and adds a novel momentum exhaustion edge.

name = "6h_WilliamsR_Extreme_1dEMA50_Volume_v1"
timeframe = "6h"
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
    
    # Calculate 6h volume median (24-period for stability, ~4 days)
    vol_median_6h = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    
    # Calculate 1d EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Williams %R, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_median_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = williams_r[i]
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 6h volume median (strict for quality)
        if vol_median_6h[i] <= 0 or np.isnan(vol_median_6h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_6h[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume spike
            if curr_wr < -80.0 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume spike
            elif curr_wr > -20.0 and downtrend and volume_confirm:
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
            # Exit: Williams %R > -50 (momentum fading) OR trend turns down
            elif curr_wr > -50.0 or not uptrend:
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
            # Exit: Williams %R < -50 (momentum fading) OR trend turns up
            elif curr_wr < -50.0 or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals