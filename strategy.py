#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Stochastic RSI with 1d volume confirmation and 1w trend filter
# - Stochastic RSI(14) for mean reversion signals: long when <0.2, short when >0.8
# - 1d volume > 1.5x 30-period average for conviction (avoids false breakouts)
# - 1w EMA(50) trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Exit on opposite Stochastic RSI extreme or trend reversal
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_StochRSI_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (30-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=30, min_periods=30).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Stochastic RSI(14) for mean reversion
    # First calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Then calculate Stochastic RSI
    rsi_min = pd.Series(rsi).rolling(window=14, min_periods=14).min()
    rsi_max = pd.Series(rsi).rolling(window=14, min_periods=14).max()
    stoch_rsi = np.divide((rsi - rsi_min), (rsi_max - rsi_min), out=np.zeros_like(rsi), where=(rsi_max - rsi_min)!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(stoch_rsi[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 6.0)
        
        if position == 0:
            # Look for long entry: uptrend (price > 1w EMA50) + oversold Stochastic RSI + volume
            if close[i] > ema_50_1w_aligned[i] and stoch_rsi[i] < 0.2 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1w EMA50) + overbought Stochastic RSI + volume
            elif close[i] < ema_50_1w_aligned[i] and stoch_rsi[i] > 0.8 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought Stochastic RSI or trend reversal
            if stoch_rsi[i] > 0.8 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold Stochastic RSI or trend reversal
            if stoch_rsi[i] < 0.2 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals