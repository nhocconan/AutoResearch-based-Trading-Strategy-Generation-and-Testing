#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mlt_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation
    # RSI extremes often reverse in ranging markets, but only trade with higher timeframe trend
    # to avoid counter-trend losses. Volume confirms momentum behind the move.
    # Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if data not ready or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) with volume + price above 4h EMA50 (uptrend)
            if rsi[i] < 30 and vol_spike[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) with volume + price below 4h EMA50 (downtrend)
            elif rsi[i] > 70 and vol_spike[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal vs 4h EMA50
            if position == 1:
                if rsi[i] > 40 or close[i] < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] < 60 or close[i] > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0