#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and session filter (08-20 UTC).
# Long when price breaks above R3 AND close > 4h EMA50 AND in active session.
# Short when price breaks below S3 AND close < 4h EMA50 AND in active session.
# Uses discrete sizing 0.20. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 15-30 trades/year on 1h timeframe (~60-120 total over 4 years).
# Camarilla levels provide intraday support/resistance; 4h EMA50 filters trend direction; session filter avoids low-volume periods.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_time = prices['open_time']
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h EMA50 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels (R3, S3) from prior day to avoid look-ahead
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Prior day's typical price (shift by 24 for 1h timeframe)
    prior_typical = pd.Series(typical_price).shift(24).values
    # Prior day's high and low (shift by 24)
    prior_high = pd.Series(high).shift(24).values
    prior_low = pd.Series(low).shift(24).values
    # Camarilla R3 and S3
    R3 = prior_typical + (prior_high - prior_low) * 1.1 / 4
    S3 = prior_typical - (prior_high - prior_low) * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and Camarilla (need 24+50 bars)
    start_idx = 74
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(R3[i]) or 
            np.isnan(S3[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        in_session = (8 <= hours[i] <= 20)
        
        # Trend filter: price vs 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND uptrend AND in session
            if curr_high > R3[i] and uptrend and in_session:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below S3 AND downtrend AND in session
            elif curr_low < S3[i] and downtrend and in_session:
                signals[i] = -0.20
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
            # Exit: price breaks below S3 OR trend turns down
            elif curr_low < S3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R3 OR trend turns up
            elif curr_high > R3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals