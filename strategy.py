#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray with 1w EMA34 trend filter.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Bull Power > 0 AND price > 1w EMA34.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND price < 1w EMA34.
# Uses discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Primary timeframe: 1d, HTF: 1w for EMA trend filter.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag and improve test generalization.

name = "1d_WilliamsAlligator_ElderRay_1wEMA34_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours (optional for 1d, but safe)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator: SMAs of median price (hlc3)
    hlc3 = (high + low + close) / 3.0
    jaw = pd.Series(hlc3).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, 8-bar shift
    teeth = pd.Series(hlc3).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, 5-bar shift
    lips = pd.Series(hlc3).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, 3-bar shift
    
    # Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 50  # warmup for Alligator (13+8), EMA13, ATR, and 1w EMA34
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (optional for 1d, keeps alignment with engine)
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Alligator alignment conditions
        bullish_alligator = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alligator = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray conditions
        bullish_elder = bull_power[i] > 0
        bearish_elder = bear_power[i] < 0
        
        # 1w EMA34 trend filter
        uptrend_1w = curr_close > ema_34_aligned[i]
        downtrend_1w = curr_close < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator AND Bullish Elder Ray AND 1w Uptrend
            if bullish_alligator and bullish_elder and uptrend_1w:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish Alligator AND Bearish Elder Ray AND 1w Downtrend
            elif bearish_alligator and bearish_elder and downtrend_1w:
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
            # Exit: Alligator turns bearish OR Elder Ray turns bearish OR 1w trend turns down
            elif (not bullish_alligator) or (not bullish_elder) or (not uptrend_1w):
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
            # Exit: Alligator turns bullish OR Elder Ray turns bullish OR 1w trend turns up
            elif (not bearish_alligator) or (not bearish_elder) or (not downtrend_1w):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals