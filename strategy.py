#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI reversal with 4h trend filter and 1d volume confirmation
# - RSI(14) on 1h for mean reversion: long when RSI < 30, short when RSI > 70
# - 4h EMA(50) trend filter: only take longs when price > 4h EMA50, shorts when price < 4h EMA50
# - 1d volume > 1.2x 20-period average for conviction
# - Exit on opposite RSI extreme (RSI > 70 for longs, RSI < 30 for shorts)
# - Session filter: only trade between 08:00-20:00 UTC to avoid low-volume periods
# - Fixed position size of 0.20 to manage risk
# - Designed to work in both bull and bear markets by following 4h trend
# - Target: ~20-30 trades/year to avoid excessive fee drift

name = "1h_RSI_4hTrend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when insufficient data
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1h volume > 1.2x 1d average volume (scaled)
        # Scale 1d average to 1h: 1d has 24x 1h bars, so divide by 24
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.2 * (vol_ma_1d_aligned[i] / 24.0)
        
        if position == 0:
            # Look for long entry: uptrend (price > 4h EMA50) + oversold RSI + volume + session
            if close[i] > ema_50_4h_aligned[i] and rsi[i] < 30 and volume_filter:
                signals[i] = 0.20
                position = 1
            # Look for short entry: downtrend (price < 4h EMA50) + overbought RSI + volume + session
            elif close[i] < ema_50_4h_aligned[i] and rsi[i] > 70 and volume_filter:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought RSI (RSI > 70)
            if rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit on oversold RSI (RSI < 30)
            if rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals