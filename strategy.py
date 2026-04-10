#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 1w EMA(50) > EMA(200) (bullish trend) AND 1d volume > 1.5x 20-bar avg
# - Short when Williams %R(14) > -20 (overbought) AND 1w EMA(50) < EMA(200) (bearish trend) AND 1d volume > 1.5x 20-bar avg
# - Exit when Williams %R returns to -50 (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Williams %R captures short-term exhaustion; 1w EMA filter ensures alignment with long-term trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Works in both bull and bear markets: mean reversion in ranges, trend filter prevents counter-trend trades

name = "1d_1w_williamsr_meanreversion_volume_trend_v1"
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
    
    # Pre-compute 1w EMA trend filter: EMA(50) vs EMA(200)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1w EMA trend to 1d timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish)
    
    # Pre-compute Williams %R(14) on 1d data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R conditions: < -80 oversold, > -20 overbought, exit at -50
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    williams_exit = np.abs(williams_r + 50) < 2.5  # Within 2.5 of -50
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(williams_oversold[i]) or np.isnan(williams_overbought[i]) or
            np.isnan(williams_exit[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long when Williams %R oversold AND 1w bullish trend AND volume spike
            if (williams_oversold[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Williams %R overbought AND 1w bearish trend AND volume spike
            elif (williams_overbought[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Williams %R = -50 (mean reversion)
            # Exit when Williams %R returns to equilibrium (-50)
            exit_signal = williams_exit[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals