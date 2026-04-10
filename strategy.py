#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 12h EMA(21) > EMA(55) (bullish trend) AND 6h volume > 1.5x 20-bar avg
# - Short when Williams %R(14) > -20 (overbought) AND 12h EMA(21) < EMA(55) (bearish trend) AND 6h volume > 1.5x 20-bar avg
# - Exit when Williams %R returns to -50 (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Williams %R captures short-term exhaustion; 12h EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Works in both bull and bear markets: mean reversion in ranges, trend filter prevents counter-trend trades
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:
        return np.zeros(n)
    
    # Pre-compute 12h EMA trend filter: EMA(21) vs EMA(55)
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_55_12h = pd.Series(close_12h).ewm(span=55, min_periods=55, adjust=False).mean().values
    ema_bullish_12h = ema_21_12h > ema_55_12h
    ema_bearish_12h = ema_21_12h < ema_55_12h
    
    # Align HTF indicators to 6h timeframe
    ema_bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_bullish_12h)
    ema_bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_bearish_12h)
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume_6h = prices['volume'].values
    volume_20_avg_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike_6h = volume_6h > (1.5 * volume_20_avg_6h)
    
    # Pre-compute Williams %R(14) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when HH == LL)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R conditions: < -80 oversold, > -20 overbought, exit at -50
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    williams_exit = np.abs(williams_r - (-50)) < 2.5  # Within 2.5 of -50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_12h_aligned[i]) or np.isnan(ema_bearish_12h_aligned[i]) or
            np.isnan(vol_spike_6h[i]) or np.isnan(williams_oversold[i]) or
            np.isnan(williams_overbought[i]) or np.isnan(williams_exit[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long when Williams %R oversold AND 12h bullish trend AND volume spike
            if (williams_oversold[i] and 
                ema_bullish_12h_aligned[i] and 
                vol_spike_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short when Williams %R overbought AND 12h bearish trend AND volume spike
            elif (williams_overbought[i] and 
                  ema_bearish_12h_aligned[i] and 
                  vol_spike_6h[i]):
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