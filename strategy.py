#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 (1d) AND 12h EMA(50) > EMA(200) (bullish trend) AND 12h volume > 1.5x 20-bar avg
# - Short when price breaks below Camarilla L3 (1d) AND 12h EMA(50) < EMA(200) (bearish trend) AND 12h volume > 1.5x 20-bar avg
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivots identify key intraday support/resistance levels; 12h EMA filter ensures alignment with intermediate trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakout strategy captures trends, pivot reversion works in ranges

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: Pivot = (H+L+C)/3, Range = H-L
    # H3 = Pivot + 1.1 * Range / 2, L3 = Pivot - 1.1 * Range / 2
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    camarilla_h3 = pivot + 1.1 * rng / 2.0
    camarilla_l3 = pivot - 1.1 * rng / 2.0
    camarilla_pivot = pivot  # exit level
    
    # Align 1d Camarilla levels to 12h timeframe (with 1-bar delay for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Pre-compute 12h EMA trend filter: EMA(50) vs EMA(200)
    close_12h = prices['close'].values  # using 12h close for EMA calc
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_bullish[i]) or
            np.isnan(ema_bearish[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Camarilla H3 AND 12h bullish trend AND volume spike
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                ema_bullish[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Camarilla L3 AND 12h bearish trend AND volume spike
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  ema_bearish[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Camarilla pivot (mean reversion)
            # Exit when price returns to Camarilla pivot point
            long_exit = prices['close'].iloc[i] < camarilla_pivot_aligned[i]
            short_exit = prices['close'].iloc[i] > camarilla_pivot_aligned[i]
            exit_signal = (position == 1 and long_exit) or (position == -1 and short_exit)
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals