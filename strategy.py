#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike confirmation
# Long: Close > Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short: Close < Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit: Price crosses 1d EMA34 OR ATR stoploss (1.5 * ATR) OR opposite Camarilla level touch (S3 for long, R3 for short)
# Camarilla pivot levels provide precise intraday support/resistance from prior 1d candle
# 1d EMA34 offers stronger trend filter than shorter HTF, reducing false signals in chop
# Volume spike confirms institutional participation in breakout
# ATR stoploss manages risk during adverse moves
# Discrete position sizing: 0.30 for long/short to balance return and drawdown
# Target: 80-140 total trades over 4 years (20-35/year) on 12h timeframe

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
    if len(df_1d) < 34:
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
    
    start_idx = 34  # warmup for EMA
    
    for i in range(start_idx, n):
        # Need prior 1d candle for Camarilla calculation (requires complete prior day)
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from prior 1d candle (using prior bar's daily OHLC)
        # We need the prior completed 1d candle's OHLC
        # Find index of prior 1d candle close in 1d data
        # Since we're on 12h timeframe, each 1d candle spans 2 bars
        prior_1d_idx = (i // 2) - 1  # prior completed 1d candle
        if prior_1d_idx < 0:
            signals[i] = 0.0
            continue
            
        # Get prior 1d OHLC from aligned 1d data (already aligned to 12h bars)
        # We'll use the HTF data directly for OHLC of prior completed 1d candle
        if prior_1d_idx >= len(df_1d):
            signals[i] = 0.0
            continue
            
        prior_high = df_1d['high'].iloc[prior_1d_idx]
        prior_low = df_1d['low'].iloc[prior_1d_idx]
        prior_close = df_1d['close'].iloc[prior_1d_idx]
        
        # Calculate Camarilla levels
        range_val = prior_high - prior_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_r3 = prior_close + range_val * 1.1 / 4
        camarilla_s3 = prior_close - range_val * 1.1 / 4
        
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
            # Stoploss: 1.5 * ATR below entry
            stop_price = entry_price - 1.5 * curr_atr
            # Exit conditions: price below 1d EMA34 OR stoploss hit OR price touches Camarilla S3 (contrarian exit)
            if (curr_close < curr_ema_1d or curr_close < stop_price or curr_close < camarilla_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: 1.5 * ATR above entry
            stop_price = entry_price + 1.5 * curr_atr
            # Exit conditions: price above 1d EMA34 OR stoploss hit OR price touches Camarilla R3 (contrarian exit)
            if (curr_close > curr_ema_1d or curr_close > stop_price or curr_close > camarilla_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: Close > Camarilla R3 AND price > 1d EMA34 AND volume spike
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: Close < Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals