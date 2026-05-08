#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_Pivot_Reversal_With_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Camarilla levels (using previous week's range)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = close_1w[0]  # avoid NaN on first bar
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    # Camarilla levels: H4, L4, H3, L3
    camarilla_h4 = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 2
    camarilla_l4 = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 2
    camarilla_h3 = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 4
    camarilla_l3 = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 4
    
    # Align Camarilla levels to daily
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Daily RSI(14)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long reversal: price touches L3 or L4 in uptrend with volume spike
            long_cond = ((close[i] <= camarilla_l3_aligned[i] * 1.002 or close[i] <= camarilla_l4_aligned[i] * 1.002) and
                        rsi[i] < 30 and  # oversold
                        ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and
                        volume_spike[i])
            
            # Short reversal: price touches H3 or H4 in downtrend with volume spike
            short_cond = ((close[i] >= camarilla_h3_aligned[i] * 0.998 or close[i] >= camarilla_h4_aligned[i] * 0.998) and
                         rsi[i] > 70 and  # overbought
                         ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches H3 or RSI > 70
            if close[i] >= camarilla_h3_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches L3 or RSI < 30
            if close[i] <= camarilla_l3_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals