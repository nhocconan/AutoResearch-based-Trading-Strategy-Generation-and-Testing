#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour Bollinger Band breakout with 4-hour trend filter, volume confirmation, and session filter (08-20 UTC)
# Long when price breaks above upper BB(20,2) on 1h, 4h close > 4h EMA50 (uptrend), volume > 1.5x 1h average volume, and time in session
# Short when price breaks below lower BB(20,2) on 1h, 4h close < 4h EMA50 (downtrend), volume > 1.5x 1h average volume, and time in session
# Exit when price returns to middle BB(20) or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h EMA50 for trend filter and 1h volume average for confirmation
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_bb_breakout_4h_ema50_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = pd.to_datetime(prices['open_time'])
    
    # Session filter: 08-20 UTC
    hours = open_time.dt.hour.values
    
    # 1h Bollinger Bands (20,2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # 1h EMA50 for exit condition
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(ema50[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or trend changes
            elif close[i] <= bb_middle[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle BB or trend changes
            elif close[i] >= bb_middle[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with BB breakout, trend alignment, volume confirmation, and session filter
            # Bullish breakout: price breaks above upper BB
            bullish_breakout = close[i] > bb_upper[i] and close[i-1] <= bb_upper[i-1]
            # Bearish breakout: price breaks below lower BB
            bearish_breakout = close[i] < bb_lower[i] and close[i-1] >= bb_lower[i-1]
            
            # Session filter: 08-20 UTC
            in_session = 8 <= hours[i] <= 20
            
            # Long: bullish breakout, 4h uptrend, volume spike, in session
            if (bullish_breakout and
                close[i] > ema50_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma[i] and
                in_session):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, 4h downtrend, volume spike, in session
            elif (bearish_breakout and
                  close[i] < ema50_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i] and
                  in_session):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals