#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + 1d RSI mean reversion + volume confirmation
# Long when price breaks above Donchian(20) upper band, RSI(14) < 50 (not overbought), volume > 1.5x 20-bar median
# Short when price breaks below Donchian(20) lower band, RSI(14) > 50 (not oversold), volume > 1.5x 20-bar median
# Exit when price returns to Donchian middle (mean of upper/lower) or RSI reverses (>70 long, <30 short)
# Designed to capture trends while avoiding overextended moves. Conservative sizing (0.25) to limit trade frequency.
# Works in bull markets (breakouts) and bear markets (mean reversion via RSI filter).

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=1).max()
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=1).min()
    donchian_mid = (highest_high + lowest_low) / 2
    
    # 1-day RSI(14)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price above Donchian upper, RSI not overbought (<70), volume spike
        if (close[i] > highest_high[i] and 
            rsi_1d_aligned[i] < 70 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price below Donchian lower, RSI not oversold (>30), volume spike
        elif (close[i] < lowest_low[i] and 
              rsi_1d_aligned[i] > 30 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to Donchian mid or RSI extreme
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= donchian_mid[i] or rsi_1d_aligned[i] >= 70)) or
               (signals[i-1] == -0.25 and (close[i] >= donchian_mid[i] or rsi_1d_aligned[i] <= 30)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_RSI1d_Volume"
timeframe = "12h"
leverage = 1.0