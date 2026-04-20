#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h volume-weighted price action with 4h trend filter
# Long when: 1h price closes above VWAP and 4h EMA50 is rising (bullish regime)
# Short when: 1h price closes below VWAP and 4h EMA50 is falling (bearish regime)
# VWAP acts as dynamic support/resistance; 4h EMA50 filters counter-trend trades
# Volume confirmation via 1h volume > 1.5x 20-period average
# Target: 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h VWAP (volume-weighted average price)
    typical_price = (prices['high'].values + prices['low'].values + prices['close'].values) / 3.0
    volume = prices['volume'].values
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative VWAP reset each day
    vwap = np.zeros(n)
    cum_num = 0.0
    cum_den = 0.0
    prev_date = None
    
    for i in range(n):
        curr_date = prices['open_time'].iloc[i].date()
        if prev_date is None or curr_date != prev_date:
            cum_num = 0.0
            cum_den = 0.0
            prev_date = curr_date
        cum_num += vwap_numerator[i]
        cum_den += vwap_denominator[i]
        if cum_den > 0:
            vwap[i] = cum_num / cum_den
        else:
            vwap[i] = typical_price[i]
    
    # Volume spike: 1h volume > 1.5x 20-period average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if outside trading session or NaN in indicators
        if not in_session[i] or np.isnan(ema50_4h_aligned[i]) or np.isnan(vwap[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # 4h trend: rising EMA50 = bullish, falling EMA50 = bearish
        if i > 0:
            ema50_rising = ema50_4h_aligned[i] > ema50_4h_aligned[i-1]
            ema50_falling = ema50_4h_aligned[i] < ema50_4h_aligned[i-1]
        else:
            ema50_rising = False
            ema50_falling = False
        
        if position == 0:
            # Long: price closes above VWAP, bullish 4h trend, volume spike
            if price > vwap[i] and ema50_rising and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price closes below VWAP, bearish 4h trend, volume spike
            elif price < vwap[i] and ema50_falling and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price closes below VWAP or bearish 4h trend
            if price < vwap[i] or ema50_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price closes above VWAP or bullish 4h trend
            if price > vwap[i] or ema50_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VWAP_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0