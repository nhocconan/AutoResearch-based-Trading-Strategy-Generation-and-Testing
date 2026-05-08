#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend and 1d momentum for direction,
# with 1h entry timing based on price action and volume confirmation.
# Uses discrete position sizing (0.20) to limit drawdown and transaction costs.
# Session filter (08-20 UTC) reduces noise. Designed to work in both bull and bear markets
# by combining trend following (4h EMA) with mean reversion (1d RSI extremes).

name = "1h_4hTrend_1dRSI_Momentum_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA21
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d momentum filter: RSI14
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_values = rsi_14_1d.values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d_values)
    
    # 1h volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(in_session[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend, 1d RSI < 30 (oversold), volume spike
            long_cond = (ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] and
                        rsi_14_1d_aligned[i] < 30 and
                        volume_spike[i])
            
            # Short: 4h downtrend, 1d RSI > 70 (overbought), volume spike
            short_cond = (ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] and
                         rsi_14_1d_aligned[i] > 70 and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend turns down OR RSI > 70 (overbought)
            if (ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] or
                rsi_14_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h trend turns up OR RSI < 30 (oversold)
            if (ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] or
                rsi_14_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals