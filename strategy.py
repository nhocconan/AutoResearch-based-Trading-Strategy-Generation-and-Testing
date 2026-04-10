#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with 1w trend filter and volume confirmation
# - Long when price breaks above upper BB(20,2) AND 1w EMA(21) > EMA(50) (bullish trend) AND 1d volume > 1.5x 20-bar avg
# - Short when price breaks below lower BB(20,2) AND 1w EMA(21) < EMA(50) (bearish trend) AND 1d volume > 1.5x 20-bar avg
# - Exit when price returns to middle BB(20) (mean reversion)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Bollinger Bands capture volatility expansion; 1w EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_bb_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(21) vs EMA(50)
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish = ema_21 > ema_50
    ema_bearish = ema_21 < ema_50
    
    # Align 1w EMA trend to 1d timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish)
    
    # Pre-compute Bollinger Bands (20,2) on 1d data
    close = prices['close'].values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    
    # BB conditions: breakout above upper, breakout below lower, exit at middle
    bb_breakout_up = close > upper_bb
    bb_breakout_down = close < lower_bb
    bb_exit = np.abs(close - middle_bb) < (0.1 * std_20)  # Within 10% of std from middle
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(bb_breakout_up[i]) or np.isnan(bb_breakout_down[i]) or
            np.isnan(bb_exit[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new BB breakout entries
            # Long when price breaks above upper BB AND 1w bullish trend AND volume spike
            if (bb_breakout_up[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below lower BB AND 1w bearish trend AND volume spike
            elif (bb_breakout_down[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to middle BB (mean reversion)
            # Exit when price returns to middle BB
            exit_signal = bb_exit[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals