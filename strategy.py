#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot long bias + 1h RSI mean reversion entries
# - Use 4h Camarilla pivot levels (H3/L3) as structural support/resistance
# - Only take longs when 1h price > 4h H3 (bullish bias) and RSI(14) < 30
# - Only take shorts when 1h price < 4h L3 (bearish bias) and RSI(14) > 70
# - Exit when price crosses 4h pivot point (mean reversion to equilibrium)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Discrete sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h (60-150 total over 4 years)

name = "4h_1h_camarilla_rsi_meanrev_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Camarilla pivots (based on previous day's high/low/close)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: H4 = C + 1.1*(H-L)/2, H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4, L4 = C - 1.1*(H-L)/2
    # Using previous bar's high/low/close for current bar's levels (no look-ahead)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]  # seed first value
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align HTF Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    # Pre-compute 1h RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(rsi[i]) or 
            not in_session[i]):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > 4h H3 AND RSI oversold
            if close[i] > camarilla_h3_aligned[i] and rsi_oversold[i]:
                position = 1
                signals[i] = 0.20
            # Short conditions: price < 4h L3 AND RSI overbought
            elif close[i] < camarilla_l3_aligned[i] and rsi_overbought[i]:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses 4h pivot point (mean reversion)
            exit_long = position == 1 and close[i] < camarilla_pivot_aligned[i]
            exit_short = position == -1 and close[i] > camarilla_pivot_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals