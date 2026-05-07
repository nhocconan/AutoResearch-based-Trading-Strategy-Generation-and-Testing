#!/usr/bin/env python3
name = "6h_Stochastic_RSI_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily RSI for trend filter
    delta_1d = pd.Series(df_1d['close']).diff()
    gain_1d = delta_1d.clip(lower=0)
    loss_1d = -delta_1d.clip(upper=0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.inf)
    rsi_1d = (100 - (100 / (1 + rs_1d))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 6h Stochastic RSI (14,14,3,3)
    rsi_period = 14
    stoch_period = 14
    k_period = 3
    d_period = 3
    
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    min_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    max_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    stoch_rsi_k = stoch_rsi.rolling(window=k_period, min_periods=k_period).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_period, min_periods=d_period).mean()
    stoch_rsi_k = stoch_rsi_k.values
    stoch_rsi_d = stoch_rsi_d.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(stoch_rsi_k[i]) or np.isnan(stoch_rsi_d[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 6h volume > 1.5x daily average volume (scaled)
        vol_6h = volume[i]
        vol_threshold = vol_ma_1d_aligned[i] / 4.0  # approximate 6h vol from daily
        
        if position == 0:
            # Long: StochRSI oversold (<20) + bullish cross + daily RSI > 50 (uptrend) + volume
            if (stoch_rsi_k[i-1] <= stoch_rsi_d[i-1] and 
                stoch_rsi_k[i] > stoch_rsi_d[i] and 
                stoch_rsi_k[i] < 20 and 
                rsi_1d_aligned[i] > 50 and 
                vol_6h > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Short: StochRSI overbought (>80) + bearish cross + daily RSI < 50 (downtrend) + volume
            elif (stoch_rsi_k[i-1] >= stoch_rsi_d[i-1] and 
                  stoch_rsi_k[i] < stoch_rsi_d[i] and 
                  stoch_rsi_k[i] > 80 and 
                  rsi_1d_aligned[i] < 50 and 
                  vol_6h > vol_threshold):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: StochRSI overbought (>80) or bearish cross
            if stoch_rsi_k[i] > 80 or (stoch_rsi_k[i] < stoch_rsi_d[i] and stoch_rsi_k[i-1] >= stoch_rsi_d[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: StochRSI oversold (<20) or bullish cross
            if stoch_rsi_k[i] < 20 or (stoch_rsi_k[i] > stoch_rsi_d[i] and stoch_rsi_k[i-1] <= stoch_rsi_d[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Stochastic RSI with daily trend filter and volume confirmation
# - Uses Stochastic RSI (14,14,3,3) on 6h for overbought/oversold signals
# - Enters on %K crossing %D in extreme zones (<20 for long, >80 for short)
# - Daily RSI >50 for long bias, <50 for short bias ensures alignment with daily trend
# - Volume confirmation: 6h volume > 1.5x average 6h volume (derived from daily)
# - Exits when StochRSI reaches opposite extreme or reversal cross occurs
# - Works in both bull (long in daily uptrend) and bear (short in daily downtrend)
# - Stochastic RSI combines momentum and mean reversion properties
# - Daily trend filter reduces whipsaws in ranging markets
# - Volume filter ensures participation during active periods
# - Position size 0.25 targets ~50-150 total trades over 4 years (12-37/year) to stay within limits
# - Novel for 6h timeframe: combines StochRSI with higher timeframe trend and volume
# - Avoids oversaturated families like pure Donchian or basic RSI strategies