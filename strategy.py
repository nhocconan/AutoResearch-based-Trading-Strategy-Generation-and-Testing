#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray power + 1d trend filter.
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND price > 1d EMA50.
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND price < 1d EMA50.
# Uses discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Primary timeframe: 6h, HTF: 1d for EMA trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours for 08-20 UTC filter (optional, reduces noise)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator: SMAs of median price (hlc3) with specific periods
    hlc3 = (high + low + close) / 3.0
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(hlc3).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift right by 8
    jaw[:8] = np.nan
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(hlc3).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift right by 5
    teeth[:5] = np.nan
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(hlc3).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 60  # warmup for Alligator and EMA
    
    for i in range(start_idx, n):
        # Optional session filter: 08-20 UTC
        # if not (8 <= hours[i] <= 20):
        #     signals[i] = 0.0
        #     continue
        
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Alligator alignment
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray power
        strong_bull_power = bull_power[i] > 0
        strong_bear_power = bear_power[i] < 0
        
        # 1d trend filter
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator AND Bull Power > 0 AND Uptrend
            if bullish_alignment and strong_bull_power and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish Alligator AND Bear Power < 0 AND Downtrend
            elif bearish_alignment and strong_bear_power and downtrend:
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
            # Exit: Alligator turns bearish OR Bear Power becomes negative OR trend turns down
            elif (not bullish_alignment or 
                  not strong_bull_power or 
                  not uptrend):
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
            # Exit: Alligator turns bullish OR Bull Power becomes positive OR trend turns up
            elif (not bearish_alignment or 
                  not strong_bear_power or 
                  not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals