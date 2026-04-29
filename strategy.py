#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla pivots: R4, R3, R2, R1, PP, S1, S2, S3, S4 calculated from prior day's OHLC
# Long: Close > R1 AND prior close <= R1 (breakout) AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short: Close < S1 AND prior close >= S1 (breakdown) AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit: ATR stoploss (2.0 * ATR) OR opposite Camarilla level touch (R3 for long, S3 for short)
# Camarilla levels work well in ranging markets common in crypto
# 1w EMA50 provides strong trend filter to avoid counter-trend trades
# Volume confirmation reduces false breakouts
# Discrete position sizing: 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe

name = "1d_Camarilla_R1S1_Breakout_1wEMA50_VolumeSpike_ATRStop_v1"
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    # R4 = Close + (High - Low) * 1.500
    # R3 = Close + (High - Low) * 1.250
    # R2 = Close + (High - Low) * 1.166
    # R1 = Close + (High - Low) * 1.083
    # PP = (High + Low + Close) / 3
    # S1 = Close - (High - Low) * 1.083
    # S2 = Close - (High - Low) * 1.166
    # S3 = Close - (High - Low) * 1.250
    # S4 = Close - (High - Low) * 1.500
    
    # We need prior day's OHLC, so shift by 1
    prior_high = np.concatenate([[np.nan], high[:-1]])
    prior_low = np.concatenate([[np.nan], low[:-1]])
    prior_close = np.concatenate([[np.nan], close[:-1]])
    
    R1 = prior_close + (prior_high - prior_low) * 1.083
    S1 = prior_close - (prior_high - prior_low) * 1.083
    R3 = prior_close + (prior_high - prior_low) * 1.250
    S3 = prior_close - (prior_high - prior_low) * 1.250
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not yet available
        if np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below stoploss OR price touches R3 (profit target/reversal)
            if curr_low <= stop_price or curr_high >= R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above stoploss OR price touches S3 (profit target/reversal)
            if curr_high >= stop_price or curr_low <= S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Prior close for breakout/breakdown confirmation
            prior_close_val = close[i-1]
            
            # Long entry: Close > R1 AND prior close <= R1 (breakout) AND price > 1w EMA50 AND volume spike
            if (curr_close > R1[i] and prior_close_val <= R1[i] and 
                curr_close > curr_ema_1w and
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Close < S1 AND prior close >= S1 (breakdown) AND price < 1w EMA50 AND volume spike
            elif (curr_close < S1[i] and prior_close_val >= S1[i] and 
                  curr_close < curr_ema_1w and
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals