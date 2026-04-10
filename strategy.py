#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion + 1w Supertrend filter + volume spike confirmation
# - Long when Williams %R(14) < -80 (oversold) AND price > 1w Supertrend (uptrend) AND volume > 1.5x 20-day avg volume
# - Short when Williams %R(14) > -20 (overbought) AND price < 1w Supertrend (downtrend) AND volume > 1.5x 20-day avg volume
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Williams %R is effective in ranging markets (2025+ bearish regime)
# - 1w Supertrend ensures we only trade with the higher timeframe trend
# - Volume spike confirms institutional participation at extremes

name = "1d_1w_williamsr_supertrend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    williams_exit_long = (williams_r > -50) & (np.roll(williams_r, 1) <= -50)  # Crossing above -50
    williams_exit_short = (williams_r < -50) & (np.roll(williams_r, 1) >= -50)  # Crossing below -50
    
    # Pre-compute 1d volume spike confirmation (>1.5x 20-day average)
    volume = prices['volume'].values
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # Pre-compute 1w Supertrend(10, 3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    
    # ATR(10)
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    supertrend = np.full_like(close_1w, np.nan)
    direction = np.full_like(close_1w, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    supertrend[9] = upper_band[9]  # seed
    direction[9] = 1  # start with uptrend assumption
    
    for i in range(10, len(close_1w)):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        if close_1w[i] <= supertrend[i-1]:
            # Trend changes to down
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            # Trend continues up
            supertrend[i] = lower_band[i]
            direction[i] = 1
    
    # Align HTF Supertrend to 1d timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    price_above_1w_supertrend = close > supertrend_aligned
    price_below_1w_supertrend = close < supertrend_aligned
    w_trend_uptrend = direction_aligned == 1
    w_trend_downtrend = direction_aligned == -1
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND 1w Supertrend uptrend AND volume spike
            if (williams_oversold[i] and w_trend_uptrend[i] and volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND 1w Supertrend downtrend AND volume spike
            elif (williams_overbought[i] and w_trend_downtrend[i] and volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when Williams %R crosses above -50
            exit_long = (position == 1 and williams_exit_long[i])
            # Exit short when Williams %R crosses below -50
            exit_short = (position == -1 and williams_exit_short[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals