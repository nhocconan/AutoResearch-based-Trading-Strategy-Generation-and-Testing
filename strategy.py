#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h/1d regime filter
# - Uses 4h EMA(34) for trend direction, 1d RSI(14) for overbought/oversold
# - 1h RSI(14) for entry timing in mean reversion zones
# - Session filter (08-20 UTC) to avoid low-liquidity hours
# - Targets 15-35 trades/year by requiring multi-timeframe alignment
# - Works in bull/bear via mean reversion in ranging markets and trend filters

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Load 1d data for regime filter (RSI)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h RSI(14) for entry timing
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.where(delta_1h > 0, delta_1h, 0)
    loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1h = avg_gain_1h / np.where(avg_loss_1h == 0, 1e-10, avg_loss_1h)
    rsi_14_1h = 100 - (100 / (1 + rs_1h))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(rsi_14_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_4h_aligned[i]
        rsi_daily = rsi_14_1d_aligned[i]
        rsi_hourly = rsi_14_1h[i]
        hour = hours[i]
        
        # Regime filters
        uptrend_regime = price > ema_trend
        downtrend_regime = price < ema_trend
        oversold = rsi_daily < 30
        overbought = rsi_daily > 70
        
        # Entry conditions with session filter
        in_session = 8 <= hour <= 20
        
        if position == 0 and in_session:
            # Long: uptrend regime + daily oversold + hourly RSI < 30
            if uptrend_regime and oversold and rsi_hourly < 30:
                signals[i] = 0.20
                position = 1
            # Short: downtrend regime + daily overbought + hourly RSI > 70
            elif downtrend_regime and overbought and rsi_hourly > 70:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 or trend breakdown
            if rsi_hourly > 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 50 or trend breakdown
            if rsi_hourly < 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_RSI_MeanReversion_Regime_v1"
timeframe = "1h"
leverage = 1.0