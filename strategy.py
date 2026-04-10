#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume spike
# - Long when Williams %R(14) < -80 (oversold) AND 1d EMA(50) > EMA(200) (bullish trend) AND volume > 1.5x 20-period average
# - Short when Williams %R(14) > -20 (overbought) AND 1d EMA(50) < EMA(200) (bearish trend) AND volume > 1.5x 20-period average
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Williams %R identifies extreme momentum exhaustion points
# - 1d EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals

name = "12h_1d_williamsr_meanreversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 1d EMAs for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d trend: bullish when EMA50 > EMA200, bearish when EMA50 < EMA200
    bullish_trend = ema_50_1d > ema_200_1d
    bearish_trend = ema_50_1d < ema_200_1d
    
    # Align HTF indicators to 12h timeframe
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend)
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND bullish 1d trend AND volume spike
            if (williams_r[i] < -80 and 
                bullish_trend_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND bearish 1d trend AND volume spike
            elif (williams_r[i] > -20 and 
                  bearish_trend_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at Williams %R = -50
            # Exit when Williams %R crosses -50 (mean reversion midpoint)
            exit_long = (position == 1 and williams_r[i] >= -50)
            exit_short = (position == -1 and williams_r[i] <= -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals