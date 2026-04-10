#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1w EMA(21) > EMA(50) AND volume > 1.5x 20-period average volume
# - Short when price breaks below Camarilla L3 level AND 1w EMA(21) < EMA(50) AND volume > 1.5x 20-period average volume
# - Exit when price crosses back inside Camarilla H3/L3 levels
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels
# - 1w EMA filter ensures we trade with the weekly trend
# - Volume confirmation reduces false breakouts

name = "1d_1w_camarilla_pivot_trend_volume_v2"
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
    
    # Pre-compute 1d OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    camarilla_h3 = np.zeros(n)
    camarilla_l3 = np.zeros(n)
    for i in range(1, n):
        camarilla_h3[i] = close[i-1] + 1.1 * (high[i-1] - low[i-1]) / 2
        camarilla_l3[i] = close[i-1] - 1.1 * (high[i-1] - low[i-1]) / 2
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = ema_21_1w > ema_50_1w
    weekly_downtrend = ema_21_1w < ema_50_1w
    
    # Align HTF indicators to 1d timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND weekly uptrend AND volume spike
            if (close[i] > camarilla_h3[i] and 
                weekly_uptrend_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND weekly downtrend AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  weekly_downtrend_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside Camarilla H3/L3 levels
            exit_long = (position == 1 and close[i] < camarilla_h3[i])
            exit_short = (position == -1 and close[i] > camarilla_l3[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals