# 4h_Bollinger_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: In low volatility (Bollinger Band squeeze), price is primed for breakout.
# Combine with daily trend filter (EMA34) and volume spike for confirmation.
# Works in both bull/bear: squeeze precedes volatility expansion in any regime.
# Target: 20-40 trades/year on 4h to avoid fee drag.
name = "4h_Bollinger_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_mult = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = sma + bb_mult * std
    lower = sma - bb_mult * std
    bb_width = (upper - lower) / sma  # normalized width
    
    # Bollinger Squeeze: width below 20-period average of width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean()
    squeeze = bb_width < bb_width_ma
    
    # Daily trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (2 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20)  # ensure indicators ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(squeeze[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout above upper band, daily uptrend, volume spike
            long_cond = (squeeze[i] and 
                        close[i] > upper[i] and
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: squeeze breakout below lower band, daily downtrend, volume spike
            short_cond = (squeeze[i] and 
                         close[i] < lower[i] and
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below SMA (mean reversion) OR opposite squeeze breakout
            if close[i] < sma[i] or (squeeze[i] and close[i] < lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above SMA OR opposite squeeze breakout
            if close[i] > sma[i] or (squeeze[i] and close[i] > upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals