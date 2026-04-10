#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# - Long when price touches Camarilla L3 support AND volume > 1.5x 20-period avg AND CHOP > 61.8 (range)
# - Short when price touches Camarilla H3 resistance AND volume > 1.5x 20-period avg AND CHOP > 61.8 (range)
# - Exit when price reaches Camarilla H4 (for longs) or L4 (for shorts) OR opposite signal
# - Uses 1w HTF trend filter: avoid longs when price < 1w EMA(50), avoid shorts when price > 1w EMA(50)
# - Discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in ranging markets via mean reversion at Camarilla levels with volume confirmation
# - Avoids trending markets via chop filter (CHOP > 61.8 = ranging)

name = "12h_1d_1w_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h typical price (for Camarilla calculation)
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3.0
    
    # Pre-compute 12h RSI(14) for exit signals
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_exit_long = (rsi > 50) & (np.roll(rsi, 1) <= 50)  # RSI crossing above 50
    rsi_exit_short = (rsi < 50) & (np.roll(rsi, 1) >= 50)  # RSI crossing below 50
    
    # Pre-compute 12h Bollinger Band Width for choppiness regime
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    chop_threshold = 0.1  # Approximately CHOP > 61.8 when BB width < 10% of middle
    chop_regime = bb_width < chop_threshold  # True when ranging (choppy)
    
    # Pre-compute 12h volume confirmation
    volume_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = prices['volume'].values > (1.5 * volume_ma)
    
    # Pre-compute 1d Camarilla pivot levels (from previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_base = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    
    # Camarilla levels
    l4 = camarilla_base - (rang * 1.1 / 2)
    l3 = camarilla_base - (rang * 1.1 / 4)
    l2 = camarilla_base - (rang * 1.1 / 6)
    l1 = camarilla_base - (rang * 1.1 / 12)
    h1 = camarilla_base + (rang * 1.1 / 12)
    h2 = camarilla_base + (rang * 1.1 / 6)
    h3 = camarilla_base + (rang * 1.1 / 4)
    h4 = camarilla_base + (rang * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    
    # Pre-compute 1w EMA(50) trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(bb_width[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price touches L3 support AND volume spike AND chopping regime AND price > 1w EMA
            long_condition = (typical_price[i] <= l3_aligned[i] * 1.001 and  # Allow small tolerance for touch
                             volume_spike[i] and 
                             chop_regime[i] and 
                             close[i] > ema_50_1w_aligned[i])
            
            # Short conditions: price touches H3 resistance AND volume spike AND chopping regime AND price < 1w EMA
            short_condition = (typical_price[i] >= h3_aligned[i] * 0.999 and  # Allow small tolerance for touch
                              volume_spike[i] and 
                              chop_regime[i] and 
                              close[i] < ema_50_1w_aligned[i])
            
            if long_condition:
                position = 1
                signals[i] = 0.25
            elif short_condition:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions
            exit_long = False
            exit_short = False
            
            if position == 1:  # Long position
                # Exit when price reaches H4 target OR RSI crosses above 50 OR opposite signal
                exit_long = (typical_price[i] >= h4_aligned[i] * 0.999 or  # Reached target
                            rsi_exit_long[i] or  # RSI mean reversion completion
                            (typical_price[i] >= l3_aligned[i] * 1.001 and  # Back above support
                             volume_spike[i] and chop_regime[i]))
            else:  # Short position
                # Exit when price reaches L4 target OR RSI crosses below 50 OR opposite signal
                exit_short = (typical_price[i] <= l4_aligned[i] * 1.001 or  # Reached target
                             rsi_exit_short[i] or  # RSI mean reversion completion
                             (typical_price[i] <= h3_aligned[i] * 0.999 and  # Back below resistance
                              volume_spike[i] and chop_regime[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals