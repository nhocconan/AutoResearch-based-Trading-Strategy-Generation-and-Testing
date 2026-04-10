#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with Williams %R mean reversion entries and 1w regime filter
# - Long when: 1d KAMA rising AND Williams %R(14) < -80 (oversold) AND 1w not in extreme overbought
# - Short when: 1d KAMA falling AND Williams %R(14) > -20 (overbought) AND 1w not in extreme oversold
# - Exit when Williams %R crosses -50 (mean reversion complete)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
# - KAMA adapts to market efficiency, Williams %R captures mean reversion in ranging markets
# - 1w regime filter prevents trading against strong weekly momentum

name = "1d_kama_williamsr_regime_v1"
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
    
    # Pre-compute Williams %R(14) on 1d
    highest_high = pd.Series(prices['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(prices['low']).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - prices['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)
    
    # Pre-compute 1d KAMA(10,2,30) - more responsive than EMA
    close_s = pd.Series(prices['close'])
    change = abs(close_s.diff(1)).values
    volatility = abs(close_s.diff(1)).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_s.values)
    kama[0] = close_s.iloc[0]
    for i in range(1, len(close_s)):
        kama[i] = kama[i-1] + sc[i] * (close_s.iloc[i] - kama[i-1])
    
    # Pre-compute 1w Williams %R for regime filter
    wh_1w = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max().values
    wl_1w = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min().values
    willr_1w = -100 * (wh_1w - df_1w['close'].values) / (wh_1w - wl_1w)
    willr_1w = np.where((wh_1w - wl_1w) == 0, -50, willr_1w)
    willr_1w_aligned = align_htf_to_ltf(prices, df_1w, willr_1w)
    
    # Pre-compute aligned 1d KAMA
    kama_aligned = align_htf_to_ltf(prices, prices, kama)  # Self-align for 1d
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(willr[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(willr_1w_aligned[i]) or i == 0):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long when KAMA rising AND oversold AND 1w not extremely overbought
            if (kama[i] > kama[i-1] and 
                willr[i] < -80 and 
                willr_1w_aligned[i] > -90):  # 1w not in extreme oversold territory
                position = 1
                signals[i] = 0.25
            # Short when KAMA falling AND overbought AND 1w not extremely oversold
            elif (kama[i] < kama[i-1] and 
                  willr[i] > -20 and 
                  willr_1w_aligned[i] < -10):  # 1w not in extreme overbought territory
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R crosses -50 (mean reversion complete)
            exit_signal = False
            if position == 1:  # Long position
                if willr[i] > -50:
                    exit_signal = True
            elif position == -1:  # Short position
                if willr[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals