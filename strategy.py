# 12H_VWAP_Trend_1wMA50_Filter
# Hypothesis: 12-hour VWAP with weekly MA50 trend filter for low-frequency, high-conviction trades.
# Long when price > VWAP and weekly MA50 rising; short when price < VWAP and weekly MA50 falling.
# VWAP provides intraday fair value; weekly MA50 filters higher timeframe trend.
# Designed for low trade frequency (<30/year) by requiring dual timeframe alignment.
# Works in bull/bear markets by following weekly trend while using 12h VWAP for entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for MA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    ma50_1w_aligned = align_htf_to_ltf(prices, df_1w, ma50_1w)
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    
    # Cumulative TP*V and cumulative volume for VWAP
    cum_tp_vol = np.cumsum(tp_vol)
    cum_vol = np.cumsum(volume)
    
    # VWAP = cumulative TP*V / cumulative volume
    vwap = np.divide(cum_tp_vol, cum_vol, out=np.zeros_like(cum_tp_vol), where=cum_vol!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for weekly MA50
        # Skip if data not ready
        if np.isnan(vwap[i]) or np.isnan(ma50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above VWAP and weekly MA50 rising
            if (close[i] > vwap[i] and 
                ma50_1w_aligned[i] > ma50_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price below VWAP and weekly MA50 falling
            elif (close[i] < vwap[i] and 
                  ma50_1w_aligned[i] < ma50_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below VWAP OR weekly MA50 turns down
                if (close[i] < vwap[i] or 
                    ma50_1w_aligned[i] < ma50_1w_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above VWAP OR weekly MA50 turns up
                if (close[i] > vwap[i] or 
                    ma50_1w_aligned[i] > ma50_1w_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_VWAP_Trend_1wMA50_Filter"
timeframe = "12h"
leverage = 1.0