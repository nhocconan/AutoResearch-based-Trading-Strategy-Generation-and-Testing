#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Uses 4h close > 4h EMA50 for uptrend, < EMA50 for downtrend as primary signal direction
# - Enters on 1h when price breaks above/below Camarilla H3/L3 levels with volume > 1.3x average
# - Exits when price retests Camarilla H4/L4 levels or volume drops below average
# - Session filter: only trade 08-20 UTC to avoid low-liquidity hours
# - Targets 15-35 trades/year (60-140 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets; trend filter adds directional bias
# - Volume confirmation prevents false breakouts in choppy conditions

name = "1h_4h_camarilla_pivot_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Pre-compute Camarilla levels on 1h data (using previous bar's high/low/close)
    # Camarilla: H4 = C + 1.1*(H-L)*1.1/2, H3 = C + 1.1*(H-L)*1.1/4, L3 = C - 1.1*(H-L)*1.1/4, L4 = C - 1.1*(H-L)*1.1/2
    high_prev = prices['high'].shift(1).values
    low_prev = prices['low'].shift(1).values
    close_prev = prices['close'].shift(1).values
    range_prev = high_prev - low_prev
    
    camarilla_h4 = close_prev + 1.1 * range_prev * 1.1 / 2.0
    camarilla_h3 = close_prev + 1.1 * range_prev * 1.1 / 4.0
    camarilla_l3 = close_prev - 1.1 * range_prev * 1.1 / 4.0
    camarilla_l4 = close_prev - 1.1 * range_prev * 1.1 / 2.0
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h4[i]) or
            np.isnan(camarilla_l4[i]) or np.isnan(volume_20_avg[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > H3 with volume spike AND 4h uptrend
            if (prices['high'].iloc[i] > camarilla_h3[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short breakdown: price < L3 with volume spike AND 4h downtrend
            elif (prices['low'].iloc[i] < camarilla_l3[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests H4/L4 levels (strong reversal signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['low'].iloc[i] < camarilla_h4[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20  # Hold long
            elif position == -1:  # Short position
                if (prices['high'].iloc[i] > camarilla_l4[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20  # Hold short
    
    return signals