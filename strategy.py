#!/usr/bin/env python3
# 1h_4h_1d_rsi_volume_regime_v1
# Hypothesis: 1h strategy using 4h RSI for trend direction, 1d ADX for regime filter, and 1h volume spike for entry timing.
# Long: 4h RSI > 50 (uptrend), 1d ADX < 25 (range/low trend), 1h RSI < 30 (oversold) + volume > 2x 20-period average.
# Short: 4h RSI < 50 (downtrend), 1d ADX < 25 (range/low trend), 1h RSI > 70 (overbought) + volume > 2x 20-period average.
# Exit: Opposite RSI condition (RSI > 70 for long exit, RSI < 30 for short exit) or ATR trailing stop (1.5x ATR).
# Uses 4h RSI for trend, 1d ADX to avoid strong trends where mean reversion fails, 1h RSI extremes for mean reversion entries.
# Target: 15-37 trades/year (60-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_volume_regime_v1"
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
    
    # 1h RSI(14) for mean reversion signals
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1h = 100 - (100 / (1 + rs))
    rsi_1h = rsi_1h.values
    
    # 1h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1h ATR(14) for volatility and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for RSI trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) for trend direction
    close_4h = pd.Series(df_4h['close'].values)
    delta_4h = close_4h.diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_4h = avg_gain_4h / avg_loss_4h
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.values
    
    # Align HTF 4h RSI to 1h timeframe (wait for completed 4h bar)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for ADX regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime detection
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = (high_1d - close_1d.shift()).abs()
    tr3_1d = (low_1d - close_1d.shift()).abs()
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_1d = high_1d.diff()
    down_1d = low_1d.shift() - low_1d
    plus_dm = up_1d.copy()
    minus_dm = down_1d.copy()
    plus_dm[up_1d <= down_1d] = 0
    minus_dm[down_1d <= up_1d] = 0
    
    # Smoothed DM
    plus_di_1d = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di_1d = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    
    # DX and ADX
    dx_1d = 100 * ((plus_di_1d - minus_di_1d).abs() / (plus_di_1d + minus_di_1d))
    adx_1d = dx_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d = adx_1d.values
    
    # Align HTF 1d ADX to 1h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(rsi_1h[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]) or
            np.isnan(rsi_4h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 1.5*ATR from high
            if long_high > 0 and close[i] < long_high - 1.5 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: RSI > 70 (overbought) or ADX > 30 (strong trend - avoid mean reversion)
            elif rsi_1h[i] > 70 or adx_1d_aligned[i] > 30:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 1.5*ATR from low
            if short_low > 0 and close[i] > short_low + 1.5 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: RSI < 30 (oversold) or ADX > 30 (strong trend - avoid mean reversion)
            elif rsi_1h[i] < 30 or adx_1d_aligned[i] > 30:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long entry: 4h RSI > 50 (uptrend), 1d ADX < 25 (low trend), 1h RSI < 30 (oversold) + volume spike
            if (rsi_4h_aligned[i] > 50 and 
                adx_1d_aligned[i] < 25 and 
                rsi_1h[i] < 30 and 
                volume_confirmed):
                position = 1
                long_high = high[i]
                signals[i] = 0.20
            # Short entry: 4h RSI < 50 (downtrend), 1d ADX < 25 (low trend), 1h RSI > 70 (overbought) + volume spike
            elif (rsi_4h_aligned[i] < 50 and 
                  adx_1d_aligned[i] < 25 and 
                  rsi_1h[i] > 70 and 
                  volume_confirmed):
                position = -1
                short_low = low[i]
                signals[i] = -0.20
    
    return signals