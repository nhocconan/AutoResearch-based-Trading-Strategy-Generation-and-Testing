#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Keltner Channel Breakout + 1w EMA Trend + Volume Spike
# Long when: 1w EMA200 uptrend + price breaks above upper Keltner Channel (EMA10 + 1.5*ATR) + volume spike
# Short when: 1w EMA200 downtrend + price breaks below lower Keltner Channel (EMA10 - 1.5*ATR) + volume spike
# Exit when: price crosses back through EMA10
# This captures trend continuation with volatility-adjusted entries, effective in both bull and bear markets.
# Target: 15-30 trades/year to minimize fee drag on daily timeframe.

name = "1d_KeltnerBreakout_1wEMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA200 for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # EMA10 for Keltner Channel middle
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR(10) for Keltner Channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel bounds
    kc_upper = ema_10 + 1.5 * atr
    kc_lower = ema_10 - 1.5 * atr
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1w EMA200 uptrend + break above upper KC + volume spike
            ema_uptrend = ema_200_1w_aligned[i] > ema_200_1w_aligned[i-1]
            breakout_up = close[i] > kc_upper[i-1]  # break above prior upper KC
            
            long_cond = ema_uptrend and breakout_up and volume_spike[i]
            
            # Short: 1w EMA200 downtrend + break below lower KC + volume spike
            ema_downtrend = ema_200_1w_aligned[i] < ema_200_1w_aligned[i-1]
            breakout_down = close[i] < kc_lower[i-1]  # break below prior lower KC
            
            short_cond = ema_downtrend and breakout_down and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA10 (trend reversal)
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA10 (trend reversal)
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals