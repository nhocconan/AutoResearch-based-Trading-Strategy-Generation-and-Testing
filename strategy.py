#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Primary timeframe: 1h for entry timing precision
# - HTF: 4h for trend direction (EMA21) and 1d for Camarilla pivot levels
# - Camarilla pivot: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
# - Entry: Long when price breaks above H4 with 4h uptrend, Short when breaks below L4 with 4h downtrend
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Discrete sizing: 0.20 to minimize fee churn and control drawdown
# - Target: 15-35 trades/year (60-140 total over 4 years) to stay within HARD MAX: 200 total

name = "1h_4h_1d_camarilla_pivot_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup for EMA21
        # Skip if any required data is invalid
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        close_price = prices['close'].iloc[i]
        open_price = prices['open'].iloc[i]
        
        if position == 0:  # Flat - look for new entries
            if in_session:
                # 4h uptrend: price > EMA21
                uptrend_4h = close_price > ema_21_4h_aligned[i]
                # 4h downtrend: price < EMA21
                downtrend_4h = close_price < ema_21_4h_aligned[i]
                
                # Long: price breaks above Camarilla H4 in uptrend
                if uptrend_4h and close_price > camarilla_h4_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below Camarilla L4 in downtrend
                elif downtrend_4h and close_price < camarilla_l4_aligned[i]:
                    position = -1
                    signals[i] = -0.20
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Outside session, stay flat
        else:  # Have position - look for exit
            # Exit conditions: reverse signal or time-based
            if position == 1:  # Long position
                # Exit: price breaks below Camarilla L4 or 4h trend turns down
                if close_price < camarilla_l4_aligned[i] or (close_price < ema_21_4h_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1, Short position
                # Exit: price breaks above Camarilla H4 or 4h trend turns up
                if close_price > camarilla_h4_aligned[i] or (close_price > ema_21_4h_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals