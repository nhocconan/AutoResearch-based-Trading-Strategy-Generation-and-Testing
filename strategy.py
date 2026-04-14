#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR-based volatility breakout with 1w RSI trend filter and volume confirmation.
# Long when price breaks above prior day's high + ATR with 1w RSI > 55 (strong uptrend) and volume > 1.5x average.
# Short when price breaks below prior day's low - ATR with 1w RSI < 45 (strong downtrend) and volume > 1.5x average.
# Exit when price returns to prior day's close or RSI crosses 50 in opposite direction.
# Uses tighter RSI thresholds (45/55) to reduce trades and improve quality. Position size 0.25 to manage drawdown.
# Designed to work in both bull and bear markets by using volatility-adjusted breakouts and RSI for trend strength.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for volatility breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volatility breakout levels: prior day's high/low ± ATR
    breakout_high = np.roll(high_1d, 1) + np.roll(atr_1d, 1)
    breakout_low = np.roll(low_1d, 1) - np.roll(atr_1d, 1)
    breakout_high[0] = np.nan
    breakout_low[0] = np.nan
    
    # Prior 1d close for exit condition
    prior_close_1d = np.roll(close_1d, 1)
    prior_close_1d[0] = np.nan
    
    # Load 1w data ONCE for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on 1w
    delta = np.diff(close_1w, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    breakout_high_aligned = align_htf_to_ltf(prices, df_1d, breakout_high)
    breakout_low_aligned = align_htf_to_ltf(prices, df_1d, breakout_low)
    prior_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_close_1d)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need breakout levels and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(breakout_high_aligned[i]) or 
            np.isnan(breakout_low_aligned[i]) or
            np.isnan(prior_close_1d_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: RSI > 55 for strong uptrend, < 45 for strong downtrend
        strong_uptrend = rsi_1w_aligned[i] > 55
        strong_downtrend = rsi_1w_aligned[i] < 45
        
        if position == 0:
            # Look for volatility breakouts with strong trend
            # Long: price breaks above breakout_high AND strong uptrend
            if (close[i] > breakout_high_aligned[i] and 
                strong_uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below breakout_low AND strong downtrend
            elif (close[i] < breakout_low_aligned[i] and 
                  strong_downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to prior 1d close or RSI crosses below 50
            if (close[i] <= prior_close_1d_aligned[i] or 
                rsi_1w_aligned[i] <= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to prior 1d close or RSI crosses above 50
            if (close[i] >= prior_close_1d_aligned[i] or 
                rsi_1w_aligned[i] >= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_VolatilityBreakout_RSI45_55_v1"
timeframe = "4h"
leverage = 1.0