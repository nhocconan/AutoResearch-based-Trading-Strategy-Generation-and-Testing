#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long: Close > Camarilla R3 AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short: Close < Camarilla S3 AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Camarilla H3/L3 midpoint OR price crosses 12h EMA50 OR ATR stoploss (2.0*ATR)
# Camarilla levels derived from previous day's range, effective in both trending and ranging markets
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# Discrete position sizing: 0.25 for long/short, 0.0 for flat to minimize fee churn

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels from previous day's OHLC
        # Need previous day's high, low, close
        day_start_idx = i - (i % 96)  # 96 4h bars per day (24h / 0.25h)
        if day_start_idx >= 96:
            prev_day_start = day_start_idx - 96
            prev_day_end = day_start_idx
            if prev_day_end <= i:  # ensure we have complete previous day
                phigh = np.max(high[prev_day_start:prev_day_end])
                plow = np.min(low[prev_day_start:prev_day_end])
                pclose = close[prev_day_end - 1]
                range_val = phigh - plow
                camarilla_h3 = pclose + range_val * 1.1 / 4
                camarilla_l3 = pclose - range_val * 1.1 / 4
                camarilla_h4 = pclose + range_val * 1.1 / 2
                camarilla_l4 = pclose - range_val * 1.1 / 2
                camarilla_h5 = pclose + range_val * 1.1
                camarilla_l5 = pclose - range_val * 1.1
                camarilla_h6 = pclose + range_val * 1.1 * 1.168
                camarilla_l6 = pclose - range_val * 1.1 * 1.168
                camarilla_h3 = camarilla_h3  # R3
                camarilla_l3 = camarilla_l3  # S3
                camarilla_h3 = camarilla_h3
                camarilla_l3 = camarilla_l3
                camarilla_mid = (camarilla_h3 + camarilla_l3) / 2.0
            else:
                signals[i] = 0.0
                continue
        else:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: Close below H3/L3 midpoint OR price below 12h EMA50 OR stoploss hit
            if curr_close < camarilla_mid or curr_close < curr_ema_12h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above H3/L3 midpoint OR price above 12h EMA50 OR stoploss hit
            if curr_close > camarilla_mid or curr_close > curr_ema_12h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Close > Camarilla R3 AND price > 12h EMA50 AND volume spike
            if (curr_close > camarilla_h3 and 
                curr_close > curr_ema_12h and
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Close < Camarilla S3 AND price < 12h EMA50 AND volume spike
            elif (curr_close < camarilla_l3 and 
                  curr_close < curr_ema_12h and
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals