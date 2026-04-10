#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Williams %R(14) from 6h data identifies overbought/oversold conditions
# - Long when %R < -80 (oversold) AND 12h EMA50 > EMA200 (uptrend) AND volume > 1.3x 20-period average
# - Short when %R > -20 (overbought) AND 12h EMA50 < EMA200 (downtrend) AND volume > 1.3x 20-period average
# - Exit when %R crosses -50 (mean reversion midpoint) or opposite signal occurs
# - Williams %R is effective in ranging markets which dominate 2025+ BTC/ETH action
# - 12h EMA filter ensures we trade mean reversion in the direction of higher timeframe trend
# - Volume confirmation prevents trading in low liquidity conditions
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years) to avoid fee drag

name = "6h_12h_williamsr_meanrev_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h EMA50 and EMA200
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = np.full_like(close_6h, np.nan, dtype=float)
    denominator = highest_high - lowest_low
    mask = denominator != 0
    williams_r[mask] = ((highest_high[mask] - close_6h[mask]) / denominator[mask]) * -100
    
    # Pre-compute 6h volume MA(20)
    volume_6h = prices['volume'].values
    volume_ma_20 = np.full_like(volume_6h, np.nan, dtype=float)
    for i in range(19, len(volume_6h)):
        volume_ma_20[i] = np.mean(volume_6h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.3x average)
        vol_spike = volume_6h[i] > 1.3 * volume_ma_20[i]
        
        williams_now = williams_r[i]
        ema_50_now = ema_50_aligned[i]
        ema_200_now = ema_200_aligned[i]
        
        # Williams %R signals
        oversold = williams_now < -80
        overbought = williams_now > -20
        mean_reversion_up = (williams_r[i-1] <= -50 and williams_now > -50)  # crosses above -50
        mean_reversion_down = (williams_r[i-1] >= -50 and williams_now < -50)  # crosses below -50
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: oversold AND 12h uptrend (EMA50 > EMA200) AND volume spike
            if (oversold and ema_50_now > ema_200_now and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: overbought AND 12h downtrend (EMA50 < EMA200) AND volume spike
            elif (overbought and ema_50_now < ema_200_now and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses -50 (mean reversion) or opposite extreme
            exit_long = (position == 1 and 
                        (mean_reversion_down or overbought))
            exit_short = (position == -1 and 
                         (mean_reversion_up or oversold))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals