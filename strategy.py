#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# - Long when price breaks above upper BB(20,2) AND 1d EMA(50) > EMA(200) AND volume > 1.5x 20-bar avg
# - Short when price breaks below lower BB(20,2) AND 1d EMA(50) < EMA(200) AND volume > 1.5x 20-bar avg
# - Exit when price returns to middle BB(20) or opposite band touch
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Bollinger squeeze captures low volatility breakouts; 1d EMA filter ensures alignment with daily trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_bb_squeeze_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1d EMA trend to 4h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    
    # Pre-compute Bollinger Bands on 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # BB(20,2): middle = SMA(20), std = 2 * stddev(20)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    
    # BB squeeze: bandwidth < 10th percentile of last 50 periods (low volatility)
    bb_width = (upper_bb - lower_bb) / middle_bb
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.1).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # BB breakout conditions
    bb_breakout_up = close > upper_bb
    bb_breakout_down = close < lower_bb
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(bb_squeeze[i]) or np.isnan(bb_breakout_up[i]) or np.isnan(bb_breakout_down[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries during squeeze
            # Long when BB squeeze AND breakout above upper BB AND 1d bullish trend AND volume spike
            if (bb_squeeze[i] and 
                bb_breakout_up[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when BB squeeze AND breakout below lower BB AND 1d bearish trend AND volume spike
            elif (bb_squeeze[i] and 
                  bb_breakout_down[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to middle BB or touches opposite band
            exit_long = close[i] <= middle_bb[i]  # Long exit: price <= middle BB
            exit_short = close[i] >= middle_bb[i]  # Short exit: price >= middle BB
            
            if (position == 1 and exit_long) or (position == -1 and exit_short):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals