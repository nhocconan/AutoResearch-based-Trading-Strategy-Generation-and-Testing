#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR(14) volatility filter
# Donchian breakouts capture strong momentum moves. 1d EMA50 ensures alignment with higher timeframe trend.
# ATR(14) > 1.5x its 50-period MA filters for sufficient volatility to avoid choppy markets.
# Discrete sizing 0.25 targets ~75-150 trades over 4 years (19-38/year) to minimize fee drag.
# Timeframe: 4h (primary timeframe for balance of signal quality and trade frequency).

name = "4h_Donchian20_1dEMA50_ATRVolFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) and its 50-period MA for volatility filter on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().shift(1).values
    vol_filter = atr > (atr_ma * 1.5)
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and ATR calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > Donchian High with 1d uptrend and volatility filter
            long_breakout = close[i] > donchian_high[i]
            ema_trend_up = close[i] > ema_50_1d_aligned[i]
            
            # Short breakdown: price < Donchian Low with 1d downtrend and volatility filter
            short_breakout = close[i] < donchian_low[i]
            ema_trend_down = close[i] < ema_50_1d_aligned[i]
            
            if long_breakout and ema_trend_up and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and ema_trend_down and vol_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian Low or trend reversal (close < EMA50)
            if close[i] < donchian_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian High or trend reversal (close > EMA50)
            if close[i] > donchian_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals