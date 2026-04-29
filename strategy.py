#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long: Close > Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short: Close < Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Camarilla midpoint (R3/S3 average) OR price crosses 1d EMA34 OR ATR stoploss
# ATR stoploss: 2.0 * ATR(14) from entry price
# Using 12h primary timeframe with 1d HTF for trend filter provides stable signals suitable for bear markets
# Volume spike confirms institutional participation and reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Discrete position sizing: 0.30 for long/short, 0.0 for flat to minimize fee churn

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "12h"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    start_idx = max(34, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels (based on previous day's OHLC)
        if i >= 28:  # Need ~28 bars of 12h data to approximate 1d OHLC
            # Approximate previous day's OHLC using last 28 bars (2*12h = 1d)
            lookback = 28
            prev_high = np.max(high[i-lookback:i])
            prev_low = np.min(low[i-lookback:i])
            prev_close = close[i-lookback]  # Close of the bar before lookback period
            
            # Camarilla levels
            range_val = prev_high - prev_low
            camarilla_h4 = prev_close + range_val * 1.1 / 2
            camarilla_l4 = prev_close - range_val * 1.1 / 2
            camarilla_h3 = prev_close + range_val * 1.1 / 4
            camarilla_l3 = prev_close - range_val * 1.1 / 4
            camarilla_mid = (camarilla_h3 + camarilla_l3) / 2.0
        else:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
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
            # Exit conditions: Close below Camarilla midpoint OR price below 1d EMA34 OR stoploss hit
            if curr_close < camarilla_mid or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above Camarilla midpoint OR price above 1d EMA34 OR stoploss hit
            if curr_close > camarilla_mid or curr_close > curr_ema_1d or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: Close > Camarilla H3 AND price > 1d EMA34 AND volume spike
            if (curr_close > camarilla_h3 and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: Close < Camarilla L3 AND price < 1d EMA34 AND volume spike
            elif (curr_close < camarilla_l3 and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals