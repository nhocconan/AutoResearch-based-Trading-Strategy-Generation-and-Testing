#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-period volume median.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-period volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Williams Alligator identifies trend initiation and continuation; 1d EMA50 filters for higher-timeframe trend alignment.
# Volume confirmation ensures breakout conviction. Works in bull markets (trend continuation) and bear markets (trend reversals).
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years).

name = "12h_WilliamsAlligator_1dEMA50_Volume_v1"
timeframe = "12h"
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
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Williams Alligator (using SMAs of median price)
    median_price = (high + low) / 2.0
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaws = pd.Series(jaws).rolling(window=8, min_periods=8).mean().values  # SMMA(13, 8)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth).rolling(window=5, min_periods=5).mean().values  # SMMA(8, 5)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips).rolling(window=3, min_periods=3).mean().values  # SMMA(5, 3)
    
    # Calculate 1d EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Alligator, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaws[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator alignment: bullish (jaws < teeth < lips) or bearish (jaws > teeth > lips)
        bullish_align = jaws[i] < teeth[i] < lips[i]
        bearish_align = jaws[i] > teeth[i] > lips[i]
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator AND uptrend AND volume spike
            if bullish_align and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: bearish Alligator AND downtrend AND volume spike
            elif bearish_align and downtrend and volume_confirm:
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
            # Exit: Alligator turns bearish OR trend turns down
            elif not bullish_align or not uptrend:
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
            # Exit: Alligator turns bullish OR trend turns up
            elif not bearish_align or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals