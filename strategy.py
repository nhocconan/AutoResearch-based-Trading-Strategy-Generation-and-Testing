#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams %R mean reversion with weekly trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND weekly EMA(21) > EMA(55) (bullish trend) AND daily volume > 1.8x 20-bar avg
# - Short when Williams %R(14) > -20 (overbought) AND weekly EMA(21) < EMA(55) (bearish trend) AND daily volume > 1.8x 20-bar avg
# - Exit when Williams %R returns to -50 (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and fee drag
# - Williams %R captures short-term exhaustion; weekly EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 20-60 trades/year on 1d timeframe (80-240 total over 4 years) - within acceptable range for 1d

name = "1d_1w_williamsr_meanreversion_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Pre-compute weekly EMA trend filter: EMA(21) vs EMA(55)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_55_1w = pd.Series(close_1w).ewm(span=55, min_periods=55, adjust=False).mean().values
    ema_bullish_1w = ema_21_1w > ema_55_1w
    ema_bearish_1w = ema_21_1w < ema_55_1w
    
    # Pre-compute daily volume confirmation: > 1.8x 20-period average
    volume_1d = prices['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_20_avg_1d)
    
    # Align HTF indicators to daily timeframe
    ema_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish_1w)
    ema_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish_1w)
    
    # Pre-compute Williams %R(14) on daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when HH == LL)
    williams_r = np.where((highest_high == lowest_low), -50, williams_r)
    
    # Williams %R conditions: < -80 oversold, > -20 overbought, exit at -50
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    williams_exit = np.abs(williams_r - (-50)) < 5  # Within 5 of -50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1w_aligned[i]) or np.isnan(ema_bearish_1w_aligned[i]) or
            np.isnan(vol_spike_1d[i]) or np.isnan(williams_oversold[i]) or
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
            # Long when Williams %R oversold AND weekly bullish trend AND volume spike
            if (williams_oversold[i] and 
                ema_bullish_1w_aligned[i] and 
                vol_spike_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short when Williams %R overbought AND weekly bearish trend AND volume spike
            elif (williams_overbought[i] and 
                  ema_bearish_1w_aligned[i] and 
                  vol_spike_1d[i]):
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