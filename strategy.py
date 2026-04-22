# 1h Mean Reversion with 4h Trend Filter and Volume Confirmation
# Hypothesis: In choppy or ranging markets, price tends to revert to the 4h VWAP. 
# Enter long when price crosses below 1h VWAP during 4h uptrend with volume confirmation.
# Enter short when price crosses above 1h VWAP during 4h downtrend with volume confirmation.
# Exit when price returns to 1h VWAP or trend reverses.
# Works in both bull and bear markets by following the 4h trend.
# Uses 1h timeframe for entry timing, 4h for trend direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h VWAP calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 4h close for trend
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: price crosses below VWAP + 4h uptrend + volume spike
            if close[i] < vwap[i] and close[i-1] >= vwap[i-1] and ema20_4h_aligned[i] > ema20_4h_aligned[i-1] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price crosses above VWAP + 4h downtrend + volume spike
            elif close[i] > vwap[i] and close[i-1] <= vwap[i-1] and ema20_4h_aligned[i] < ema20_4h_aligned[i-1] and vol_spike:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to VWAP or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above VWAP or 4h trend turns down
                if close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                    exit_signal = True
                elif ema20_4h_aligned[i] < ema20_4h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below VWAP or 4h trend turns up
                if close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                    exit_signal = True
                elif ema20_4h_aligned[i] > ema20_4h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_VWAP_MeanReversion_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0