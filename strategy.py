#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Long when price breaks above H3 level with 4h uptrend (EMA21 > EMA50) during 08-20 UTC
# - Short when price breaks below L3 level with 4h downtrend (EMA21 < EMA50) during 08-20 UTC
# - Uses Camarilla levels from previous 4h bar to avoid look-ahead
# - Position size 0.20 to limit drawdown
# - Session filter reduces noise and overtrading
# - Designed for 15-30 trades/year to avoid fee drag

name = "1h_4h_camarilla_breakout_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(21) and EMA(50) for trend filter
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous 4h bar Camarilla levels (H3, L3)
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    camarilla_h3_4h = close_4h + (1.1 * (high_4h - low_4h) / 2)
    camarilla_l3_4h = close_4h - (1.1 * (high_4h - low_4h) / 2)
    camarilla_h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_h3_4h_aligned[i]) or np.isnan(camarilla_l3_4h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above H3 with 4h uptrend
            if (prices['close'].iloc[i] > camarilla_h3_4h_aligned[i] and 
                ema_21_4h_aligned[i] > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short signal: price breaks below L3 with 4h downtrend
            elif (prices['close'].iloc[i] < camarilla_l3_4h_aligned[i] and 
                  ema_21_4h_aligned[i] < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
        else:  # Have position - look for exit
            # Exit when price returns to opposite Camarilla level or trend reverses
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < camarilla_l3_4h_aligned[i] or 
                    ema_21_4h_aligned[i] < ema_50_4h_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # Short position
                if (prices['close'].iloc[i] > camarilla_h3_4h_aligned[i] or 
                    ema_21_4h_aligned[i] > ema_50_4h_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals