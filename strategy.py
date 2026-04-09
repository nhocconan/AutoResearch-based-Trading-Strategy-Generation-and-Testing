#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d regime filter
# - Primary timeframe: 1h for entry timing
# - HTF: 4h for trend direction (EMA21), 1d for volatility regime (ATR ratio)
# - Long when: price breaks above H3 Camarilla pivot level AND 4h EMA21 > 1h EMA21 (uptrend) AND 1d ATR(14)/ATR(50) < 0.8 (low vol)
# - Short when: price breaks below L3 Camarilla pivot level AND 4h EMA21 < 1h EMA21 (downtrend) AND 1d ATR(14)/ATR(50) < 0.8 (low vol)
# - Uses discrete position size 0.20 to minimize fee churn
# - ATR-based stoploss: exit when price moves 2.0x ATR against position
# - Session filter: 08-20 UTC to avoid Asian session noise
# - Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe

name = "1h_4h_1d_camarilla_pivot_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Session filter: 08-20 UTC (pre-compute before loop)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend direction
    close_4h = df_4h['close'].values
    ema_4h_21 = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_21_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_21)
    
    # Calculate 1h EMA21 for trend confirmation
    close_s = pd.Series(close)
    ema_1h_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # First bar
    
    atr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # Ratio < 0.8 indicates low volatility regime
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1h ATR(14) for stoploss
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    tr_h[0] = tr1_h[0]  # First bar
    atr_1h = pd.Series(tr_h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h Camarilla pivot levels (based on previous bar)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #          L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_ = prev_high - prev_low
    camarilla_multiplier = 1.1 * range_ / 4  # For H3/L3
    
    h3 = prev_close + camarilla_multiplier  # H3 level
    l3 = prev_close - camarilla_multiplier  # L3 level
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(ema_4h_21_aligned[i]) or np.isnan(ema_1h_21[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(atr_1h[i]) or
            atr_1h[i] <= 0 or atr_ratio_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime filters
        uptrend_4h = ema_4h_21_aligned[i] > ema_1h_21[i]  # 4h EMA above 1h EMA
        downtrend_4h = ema_4h_21_aligned[i] < ema_1h_21[i]  # 4h EMA below 1h EMA
        low_volatility = atr_ratio_aligned[i] < 0.8  # 1d ATR(14)/ATR(50) < 0.8
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.0x ATR from highest high
            if close[i] < highest_high_since_entry - 2.0 * atr_1h[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.0 * atr_1h[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if low_volatility:  # Only trade in low volatility regime
                # Long entry: price breaks above H3 AND uptrend on 4h
                if close[i] > h3[i] and uptrend_4h:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.20
                # Short entry: price breaks below L3 AND downtrend on 4h
                elif close[i] < l3[i] and downtrend_4h:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.20
    
    return signals