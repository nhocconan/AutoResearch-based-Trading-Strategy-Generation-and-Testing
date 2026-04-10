#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 12h EMA(21) > EMA(50) (uptrend) AND volume > 1.5x 20-period average
# - Short when Williams %R(14) > -20 (overbought) AND 12h EMA(21) < EMA(50) (downtrend) AND volume > 1.5x 20-period average
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R identifies extreme momentum conditions that often reverse
# - 12h EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_12h_williamsr_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h EMA trend filter
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_trend_up = ema_21_12h > ema_50_12h
    ema_trend_down = ema_21_12h < ema_50_12h
    
    # Align HTF indicators to 4h timeframe
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_up)
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_down)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_trend_up_aligned[i]) or np.isnan(ema_trend_down_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND 12h uptrend AND volume spike
            if (williams_r[i] < -80 and 
                ema_trend_up_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND 12h downtrend AND volume spike
            elif (williams_r[i] > -20 and 
                  ema_trend_down_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            exit_long = (position == 1 and williams_r[i] > -50)
            exit_short = (position == -1 and williams_r[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals