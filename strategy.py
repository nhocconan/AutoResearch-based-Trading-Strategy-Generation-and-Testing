#!/usr/bin/env python3
"""
Experiment #2294: 1h RSI(14) pullback + 4h/1d EMA trend filter + volume spike + session filter
HYPOTHESIS: In strong 4h/1d trends (EMA20 > EMA50), wait for 1h RSI to pull back to 40-60 during 08-20 UTC session, then enter on volume spike (>2x 20-bar avg) in trend direction. Uses discrete sizing (0.20) to minimize fee churn. Designed for 1h timeframe with low trade frequency (target: 60-150 total trades over 4 years) to overcome fee drag in difficult 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2294_1h_rsi_pullback_4h_1d_ema_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h EMA20/EMA50 for trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h = np.where(ema20_4h > ema50_4h, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d EMA20/EMA50 for stronger trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = np.where(ema20_1d > ema50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: RSI(14), Volume MA(20) ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size - conservative for 1h timeframe
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # Skip if any indicator not ready
        if (np.isnan(rsi[i]) or np.isnan(trend_4h_aligned[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Exit logic: close position on opposite volume spike or session end
        if in_position:
            # Exit on volume spike in opposite direction OR when leaving session
            vol_spike = vol_ratio[i] > 2.0
            next_hour = hours[i+1] if i+1 < n else -1
            next_in_session = 8 <= next_hour <= 20 if i+1 < n else False
            
            if position_side > 0:  # Long position
                if vol_spike and trend_4h_aligned[i] < 0:  # Volume spike against trend
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif not next_in_session:  # About to leave session
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if vol_spike and trend_4h_aligned[i] > 0:  # Volume spike against trend
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif not next_in_session:  # About to leave session
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # New position entry: RSI pullback in trend direction with volume spike
        # Require both 4h and 1d trend alignment for stronger filter
        if (trend_4h_aligned[i] > 0 and trend_1d_aligned[i] > 0):  # Both timeframes uptrend
            # Long: RSI pullback to 40-60 (not oversold) + volume spike
            if 40 <= rsi[i] <= 60 and vol_ratio[i] > 2.0:
                in_position = True
                position_side = 1
                signals[i] = SIZE
        elif (trend_4h_aligned[i] < 0 and trend_1d_aligned[i] < 0):  # Both timeframes downtrend
            # Short: RSI pullback to 40-60 (not overbought) + volume spike
            if 40 <= rsi[i] <= 60 and vol_ratio[i] > 2.0:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals