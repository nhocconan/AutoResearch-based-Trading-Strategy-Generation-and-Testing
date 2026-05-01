#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w EMA50 trend filter + volume confirmation.
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND price > 1w EMA50 AND volume > 1.5x 12h volume average.
# Short when Alligator jaws > teeth > lips AND price < 1w EMA50 AND volume > 1.5x 12h volume average.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Alligator calculated from completed 12h bars to avoid look-ahead.
# Volume spike filters low-momentum signals. 1w EMA50 ensures trades only in established trends.
# Works in bull (Alligator alignment with uptrend) and bear (Alligator alignment with downtrend).
# Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years).

name = "12h_WilliamsAlligator_1wEMA50_Volume_v1"
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
    
    # Load 12h data ONCE before loop for Alligator (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator (SMMA = Smoothed Moving Average)
    close_12h = df_12h['close'].values
    # Jaws: 13-period SMMA, 8 bars ahead
    jaws_12h = pd.Series(close_12h).ewm(alpha=1/13, adjust=False).mean().values
    jaws_12h = np.roll(jaws_12h, 8)
    # Teeth: 8-period SMMA, 5 bars ahead
    teeth_12h = pd.Series(close_12h).ewm(alpha=1/8, adjust=False).mean().values
    teeth_12h = np.roll(teeth_12h, 5)
    # Lips: 5-period SMMA, 3 bars ahead
    lips_12h = pd.Series(close_12h).ewm(alpha=1/5, adjust=False).mean().values
    lips_12h = np.roll(lips_12h, 3)
    
    # Align Alligator lines to 12h timeframe (wait for completed 12h bar)
    jaws_12h_aligned = align_htf_to_ltf(prices, df_12h, jaws_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Load 1w data ONCE before loop for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Alligator, EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaws_12h_aligned[i]) or 
            np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 1.5x 12h volume average (moderate to reduce trades)
        # Use 20-period volume average from 12h data
        if i >= 20:
            vol_ma_12h = np.mean(volume[max(0, i-19):i+1])
            volume_spike = curr_volume > (vol_ma_12h * 1.5)
        else:
            volume_spike = False
        
        # Alligator alignment conditions
        alligator_long = (jaws_12h_aligned[i] < teeth_12h_aligned[i]) and (teeth_12h_aligned[i] < lips_12h_aligned[i])
        alligator_short = (jaws_12h_aligned[i] > teeth_12h_aligned[i]) and (teeth_12h_aligned[i] > lips_12h_aligned[i])
        
        # Trend filter: price vs 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator alignment (jaws<teeth<lips) AND uptrend AND volume spike
            if (alligator_long and 
                uptrend and 
                volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Alligator alignment (jaws>teeth>lips) AND downtrend AND volume spike
            elif (alligator_short and 
                  downtrend and 
                  volume_spike):
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
            # Exit: Alligator alignment breaks OR trend turns down
            elif not (jaws_12h_aligned[i] < teeth_12h_aligned[i] and teeth_12h_aligned[i] < lips_12h_aligned[i]) or (not uptrend):
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
            # Exit: Alligator alignment breaks OR trend turns up
            elif not (jaws_12h_aligned[i] > teeth_12h_aligned[i] and teeth_12h_aligned[i] > lips_12h_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals