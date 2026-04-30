#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla levels act as intraday support/resistance; breakouts signal momentum.
# EMA34 provides trend bias to avoid counter-trend trades; volume confirms breakout strength.
# Long: price > R3 AND price > EMA34 (bullish) AND volume spike
# Short: price < S3 AND price < EMA34 (bearish) AND volume spike
# ATR-based stoploss: exit when price moves against position by 2.5 * ATR(14)
# Discrete sizing 0.25 to control risk and minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull via breakout longs, in bear via breakout shorts during rallies.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d EMA(34) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 14)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla levels for current 12h bar using prior 12h bar's OHLC
        if i >= 1:
            # Use previous bar's OHLC to calculate today's Camarilla levels (no look-ahead)
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            rang = prev_high - prev_low
            if rang <= 0:
                # Avoid division by zero; skip signal generation if range is invalid
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
                continue
            
            # Camarilla levels
            R3 = prev_close + rang * 1.1 / 4
            S3 = prev_close - rang * 1.1 / 4
            
            # Volume confirmation: volume > 1.5x 20-period average (calculated on prior bars)
            if i >= 20:
                vol_ma_20 = np.mean(volume[i-20:i])
                volume_spike = volume[i] > (1.5 * vol_ma_20)
            else:
                volume_spike = False
        else:
            # Not enough data for Camarilla calculation
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price > R3 AND price > EMA34 (bullish bias)
                if (curr_close > R3 and 
                    curr_close > curr_ema_34_1d):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price < S3 AND price < EMA34 (bearish bias)
                elif (curr_low < S3 and 
                      curr_close < curr_ema_34_1d):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # ATR-based stoploss: exit when price drops below entry - 2.5 * ATR
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # ATR-based stoploss: exit when price rises above entry + 2.5 * ATR
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals