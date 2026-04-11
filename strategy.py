#!/usr/bin/env python3
# 1h_4h_1d_rsi_momentum_v1
# Strategy: 1-hour RSI momentum with 4-hour trend filter and 1-day volatility regime
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: RSI(14) captures short-term momentum while 4h EMA50 filters trend direction and 1d ATR percentile avoids choppy markets.
# Long: RSI crosses above 50 with 4h uptrend and low volatility regime (ATR < 50th percentile).
# Short: RSI crosses below 50 with 4h downtrend and low volatility regime.
# Uses volatility regime filter to reduce whipsaws in ranging markets, targeting 15-30 trades/year.
# Volatility filter based on 1d ATR(14) percentile to distinguish trending vs ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Calculate 50th percentile of ATR for regime detection (using expanding window to avoid look-ahead)
    atr_percentile = pd.Series(atr_14).expanding(min_periods=30).quantile(0.50).values
    atr_low_regime = atr_14 < atr_percentile  # Low volatility regime
    atr_low_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_low_regime, additional_delay_bars=0)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_low_regime_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # RSI centerline crosses
        rsi_cross_up = rsi[i-1] < 50 and rsi[i] >= 50
        rsi_cross_down = rsi[i-1] > 50 and rsi[i] <= 50
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volatility regime: low volatility (trending market)
        low_vol = atr_low_regime_aligned[i]
        
        # Session filter
        in_session = session_filter[i]
        
        # Entry logic: RSI cross + trend + low vol + session
        if rsi_cross_up and uptrend and low_vol and in_session and position != 1:
            position = 1
            signals[i] = 0.20
        elif rsi_cross_down and downtrend and low_vol and in_session and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: opposite RSI cross with volume confirmation (using price action as proxy)
        elif position == 1 and rsi_cross_down and in_session:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi_cross_up and in_session:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals