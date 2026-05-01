#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter.
# Uses Alligator (jaw/teeth/lips) to identify trend absence/presence and Elder Ray (bull/bear power) for entry timing.
# Long when: Elder Bull Power > 0 AND price > Alligator Teeth AND 1d close > 1d EMA50 (uptrend)
# Short when: Elder Bear Power < 0 AND price < Alligator Teeth AND 1d close < 1d EMA50 (downtrend)
# Uses discrete sizing 0.25. ATR(21) stop: signal→0 when price moves against position by 3.0*ATR.
# Target: 50-100 total trades over 4 years (12-25/year) on 6h timeframe.
# Alligator filters choppy markets, Elder Ray provides momentum entries, 1d EMA50 ensures higher timeframe trend alignment.

name = "6h_Alligator_ElderRay_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate ATR(21) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    # Williams Alligator on 6h (SMAs with specific periods)
    jaw_period, teeth_period, lips_period = 13, 8, 5
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Alligator, Elder Ray, ATR, and 1d EMA
    start_idx = max(jaw_period, teeth_period, lips_period, 21, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Alligator trend detection: intertwined = chop, separated = trend
        # Jaw > Teeth > Lips = uptrend, Lips > Teeth > Jaw = downtrend
        alligator_uptrend = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        alligator_downtrend = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        
        # 1d trend filter
        day_uptrend = curr_close > ema_50_1d_aligned[i]
        day_downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Elder Bull Power positive AND price > Alligator Teeth AND 1d uptrend
            if (bull_power[i] > 0) and (curr_close > teeth[i]) and day_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Elder Bear Power negative AND price < Alligator Teeth AND 1d downtrend
            elif (bear_power[i] < 0) and (curr_close < teeth[i]) and day_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 3.0*ATR
            if curr_close < entry_price - 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Bear Power negative OR price < Alligator Lips OR 1d trend turns down
            elif (bear_power[i] < 0) or (curr_close < lips[i]) or (not day_uptrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 3.0*ATR
            if curr_close > entry_price + 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Bull Power positive OR price > Alligator Lips OR 1d trend turns up
            elif (bull_power[i] > 0) or (curr_close > lips[i]) or (not day_downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals