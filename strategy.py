#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and ATR-based volatility filter.
# Uses 1d Donchian channel breakouts for trend continuation, filtered by 1w EMA34 trend direction.
# ATR-based volatility filter ensures trades only occur during sufficient volatility regimes.
# Works in both bull (buy upper band with uptrend) and bear (sell lower band with downtrend).
# Discrete position sizing (0.25) balances return and drawdown. Target: 30-100 trades over 4 years.

name = "1d_Donchian20_Breakout_1wEMA34_ATRFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for volatility filter on 1d timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Donchian channel (20-period)
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, donchian_period) + 1  # 35
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: 1w EMA34 direction
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Volatility filter: ATR > 0.5% of price (ensures sufficient volatility)
        volatility_filter = atr_14[i] > (curr_close * 0.005)
        
        # Donchian breakout conditions
        breakout_up = curr_close > upper_channel[i-1]  # Break above previous upper band
        breakout_down = curr_close < lower_channel[i-1]  # Break below previous lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper channel AND uptrend AND volatility filter
            if breakout_up and uptrend and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower channel AND downtrend AND volatility filter
            elif breakout_down and downtrend and volatility_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below lower channel (reversal signal)
            if curr_close < lower_channel[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above upper channel (reversal signal)
            if curr_close > upper_channel[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals