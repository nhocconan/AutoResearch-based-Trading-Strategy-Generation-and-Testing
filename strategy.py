#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and 1d volume spike confirmation.
# Long when RSI < 30 (oversold) AND price > 4h EMA50 (uptrend) AND 1d volume > 2.0x 20-day average.
# Short when RSI > 70 (overbought) AND price < 4h EMA50 (downtrend) AND 1d volume > 2.0x 20-day average.
# Uses discrete sizing 0.20. Session filter: 08-20 UTC to reduce noise.
# RSI mean reversion works in ranging markets; EMA50 filters counter-trend trades; volume confirms institutional interest.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years).

name = "1h_RSI14_MeanRev_4hEMA50_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data ONCE before loop for EMA50 (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume average (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Start after warmup for RSI, EMA, and volume
    start_idx = 50
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current 1h volume > 2.0x 1d volume average (scaled)
        # Scale 1d average to approximate 1h expectation: 1d vol / 24
        vol_ma_1h_equiv = vol_ma_20_1d_aligned[i] / 24.0
        if vol_ma_1h_equiv <= 0 or np.isnan(vol_ma_1h_equiv):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_1h_equiv * 2.0)
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Trend filter: price vs 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: RSI oversold AND uptrend AND volume confirmation
            if (rsi_oversold and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought AND downtrend AND volume confirmation
            elif (rsi_overbought and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: RSI returns to neutral (50) OR trend turns down
            if rsi[i] >= 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) OR trend turns up
            if rsi[i] <= 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals