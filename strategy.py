#37: 1d_VWAP_Trend_Confirm_15mBreakout_v1
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 15m VWAP breakout confirmed by 1d trend and volume spike.
# Uses 15m price crossing VWAP with 1d EMA50 trend filter and volume > 2x average.
# Designed for 50-100 trades/year to balance opportunity and fee drag.
# Works in bull markets (breakouts with trend) and bear markets (fades from VWAP with trend).
name = "15m_VWAP_Trend_Confirm_15mBreakout_v1"
timeframe = "15m"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_15m = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate VWAP for 15m (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).cumsum().values
    vwap_den = pd.Series(volume).cumsum().values
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_15m[i]) or np.isnan(vwap[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Price crosses above VWAP with 1d uptrend and volume spike
            if close[i] > vwap[i] and close[i] > ema50_15m[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below VWAP with 1d downtrend and volume spike
            elif close[i] < vwap[i] and close[i] < ema50_15m[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below VWAP OR 1d trend turns down
            if close[i] < vwap[i] or close[i] < ema50_15m[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above VWAP OR 1d trend turns up
            if close[i] > vwap[i] or close[i] > ema50_15m[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals