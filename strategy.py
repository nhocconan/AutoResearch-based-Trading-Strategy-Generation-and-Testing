#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long: Close > Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short: Close < Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Camarilla H3/L3 midpoint OR price crosses 1d EMA34 OR ATR stoploss (2.0)
# Camarilla levels provide intraday support/resistance that works in ranging and trending markets
# EMA34 filter ensures alignment with higher timeframe trend to avoid whipsaws
# Volume spike confirms institutional participation
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
    
    start_idx = max(50, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels (based on previous bar's OHLC)
        if i >= 1:
            # Previous bar's OHLC
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_open = prices['open'].iloc[i-1] if 'open' in prices.columns else close[i-1]
            
            # Camarilla levels calculation
            range_val = prev_high - prev_low
            camarilla_h3 = prev_close + range_val * 1.1 / 4
            camarilla_l3 = prev_close - range_val * 1.1 / 4
            camarilla_h4 = prev_close + range_val * 1.1 / 2
            camarilla_l4 = prev_close - range_val * 1.1 / 2
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
            # Exit conditions: Close below Camarilla H3/L3 midpoint OR price below 1d EMA34 OR stoploss hit
            camarilla_mid = (camarilla_h3 + camarilla_l3) / 2.0
            if curr_close < camarilla_mid or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above Camarilla H3/L3 midpoint OR price above 1d EMA34 OR stoploss hit
            camarilla_mid = (camarilla_h3 + camarilla_l3) / 2.0
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