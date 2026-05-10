#!/usr/bin/env python3
# 4h_Stochastic_RSI_Breakout
# Hypothesis: Stochastic RSI identifies overbought/oversold conditions during low volatility squeezes.
# A breakout from a Bollinger Bandwidth squeeze with Stochastic RSI crossing above 80 (overbought) or below 20 (oversold)
# and volume confirmation signals a strong directional move. Uses 1d EMA trend filter for higher reliability.
# Designed for low trade frequency (20-40/year) to minimize fee drift.

name = "4h_Stochastic_RSI_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bandwidth for squeeze detection (20, 2)
    bb_period = 20
    bb_mult = 2
    close_series = pd.Series(close)
    bb_ma = close_series.ewm(span=bb_period, adjust=False, min_periods=bb_period).mean()
    bb_std = close_series.ewm(span=bb_period, adjust=False, min_periods=bb_period).std()
    bb_upper = bb_ma + bb_mult * bb_std
    bb_lower = bb_ma - bb_mult * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma  # Normalized bandwidth
    
    # Bollinger Bandwidth rank (50-period) to identify squeeze (<20th percentile)
    bbw_series = pd.Series(bb_width.values)
    bbw_rank = bbw_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    squeeze_condition = bbw_rank < 0.2  # Below 20th percentile = low volatility squeeze
    
    # Stochastic RSI (14, 14, 3, 3)
    rsi_period = 14
    stoch_period = 14
    k_smooth = 3
    d_smooth = 3
    
    # RSI calculation
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_min = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
    # Replace division by zero with 50 (neutral)
    stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], 50)
    
    # Smooth K and D
    k = stoch_rsi.ewm(alpha=1/k_smooth, adjust=False, min_periods=k_smooth).mean()
    d = k.ewm(alpha=1/d_smooth, adjust=False, min_periods=d_smooth).mean()
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 50)  # Enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or \
           np.isnan(k.iloc[i]) or np.isnan(d.iloc[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i]) or \
           np.isnan(squeeze_condition.iloc[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Extract values
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        k_val = k.iloc[i]
        d_val = d.iloc[i]
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        squeeze_val = squeeze_condition.iloc[i]
        trend_filter = ema_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper BB, Stochastic RSI K crosses above D (bullish crossover),
            # during squeeze, with volume confirmation and above 1d EMA
            if (close[i] > bb_up and 
                k_val > d_val and k_val < 80 and  # Avoid extreme overbought
                squeeze_val and vol_confirm and 
                close[i] > trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB, Stochastic RSI K crosses below D (bearish crossover),
            # during squeeze, with volume confirmation and below 1d EMA
            elif (close[i] < bb_low and 
                  k_val < d_val and k_val > 20 and  # Avoid extreme oversold
                  squeeze_val and vol_confirm and 
                  close[i] < trend_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below middle BB OR Stochastic RSI crosses below 50 (momentum loss)
            bb_mid = (bb_up + bb_low) / 2
            if close[i] < bb_mid or k_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above middle BB OR Stochastic RSI crosses above 50
            bb_mid = (bb_up + bb_low) / 2
            if close[i] > bb_mid or k_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals