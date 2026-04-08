# 1d_vwap_mean_reversion_v1
# Hypothesis: Mean reversion at VWAP with 1-week trend filter. VWAP acts as dynamic support/resistance.
# Long when price crosses above VWAP in weekly uptrend with volume confirmation.
# Short when price crosses below VWAP in weekly downtrend with volume confirmation.
# Uses 1d timeframe to reduce trade frequency and capture mean reversion moves.
# Weekly trend filter prevents counter-trend trades in strong trends.
# Volume surge confirms institutional participation at VWAP retests.
# Target: 10-25 trades/year to minimize fee drag while capturing meaningful reversions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_vwap_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP: cumulative (price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Volume filter: 1.3x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.3 * vol_ma[i]
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend direction
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = close_1w > ema_21_1w
    weekly_downtrend = close_1w < ema_21_1w
    
    # Align weekly trend to daily timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 1)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below VWAP or weekly trend turns down
            if close[i] < vwap[i] or weekly_downtrend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above VWAP or weekly trend turns up
            if close[i] > vwap[i] or weekly_uptrend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price crosses above VWAP, weekly uptrend, volume surge
            if (close[i] > vwap[i] and 
                weekly_uptrend_aligned[i] > 0.5 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price crosses below VWAP, weekly downtrend, volume surge
            elif (close[i] < vwap[i] and 
                  weekly_downtrend_aligned[i] > 0.5 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals